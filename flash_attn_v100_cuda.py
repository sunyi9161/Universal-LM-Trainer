# -*- coding: utf-8 -*-
"""
flash_attn_v100_cuda.py — 真·FlashAttention-2 on V100 (sm70) 的 Python 封装。

通过 torch.utils.cpp_extension 现场编译 `csrc/flash_attn_v100/flash_attn_v100.cu`
（nvcc 目标 sm_70，使用 Volta 张量核心 HMMA.884），编译产物缓存在
`csrc/flash_attn_v100/build/`，第一次约 1-3 分钟，之后直接复用。

对外接口与 PyTorch SDPA 一致：
    flash_attn_v100(q, k, v, dropout_p=0.0, causal=True, softmax_scale=None)
    q, k, v: (B, H, S, D) fp16, D 必须为 64, 最后一维连续
    返回: (B, H, S, D) fp16
"""

import itertools
import os

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, 'csrc', 'flash_attn_v100')
_BUILD_DIR = os.path.join(_SRC_DIR, 'build')

_EXT = None
_BUILD_ERROR = None

# 反向必须复用前向的 dropout mask。mask 是 (seed, row, col, bh) 的纯函数，
# 所以只要把前向用的 seed 存进 ctx 就够了，不需要保存 N x N 的 mask。
_SEED_COUNTER = itertools.count(1)


def _next_seed():
    return (int(torch.cuda.initial_seed()) * 2654435761 + next(_SEED_COUNTER)) & 0xFFFFFFFF


def _build():
    global _EXT, _BUILD_ERROR
    if _EXT is not None or _BUILD_ERROR is not None:
        return
    try:
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.0')
        from torch.utils.cpp_extension import load
        os.makedirs(_BUILD_DIR, exist_ok=True)
        _EXT = load(
            name='flash_attn_v100_cuda',
            sources=[os.path.join(_SRC_DIR, 'flash_attn_v100.cu'),
                     os.path.join(_SRC_DIR, 'flash_attn_v100_bwd.cu'),
                     os.path.join(_SRC_DIR, 'fast_ops.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
            extra_cflags=['-O3'],
            build_directory=_BUILD_DIR,
            verbose=False,
        )
    except Exception as e:  # pragma: no cover - 编译失败时给出可读原因
        _BUILD_ERROR = e
        _EXT = None


def is_available():
    _build()
    return _EXT is not None


def build_error():
    _build()
    return _BUILD_ERROR


class _FlashAttnV100Cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, causal, softmax_scale):
        _build()
        if _EXT is None:
            raise RuntimeError(f'flash_attn_v100 CUDA 扩展未编译成功: {_BUILD_ERROR}')
        seed = _next_seed()
        o, lse = _EXT.fa_fwd(q, k, v, float(dropout_p), bool(causal),
                             float(softmax_scale), int(seed))
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.dropout_p = float(dropout_p)
        ctx.causal = bool(causal)
        ctx.softmax_scale = float(softmax_scale)
        ctx.seed = int(seed)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        _build()
        if not hasattr(_EXT, 'fa_bwd'):
            raise NotImplementedError('flash_attn_v100: sm70 反向 kernel 尚未实现')
        dq, dk, dv = _EXT.fa_bwd(q, k, v, o, lse, do.contiguous(),
                                 ctx.dropout_p, ctx.causal, ctx.softmax_scale,
                                 ctx.seed, 7)
        return dq, dk, dv, None, None, None


def _fa_ready(t):
    """kernel 只接受 fp16、(B,H,S,64)、最后一维连续、各维跨度 32B 对齐的张量。"""
    return (t.is_cuda and t.dtype == torch.float16 and t.dim() == 4
            and t.size(3) == 64 and t.stride(3) == 1
            and t.data_ptr() % 32 == 0
            and all((s * 2) % 32 == 0 for s in t.stride()[:3]))


def can_use(q, k, v):
    """能否直接喂给 sm70 kernel（否则调用方应回退到 SDPA）。"""
    return (_fa_ready(q) and _fa_ready(k) and _fa_ready(v)
            and q.shape == k.shape == v.shape)


class _RopeV100Cuda(torch.autograd.Function):
    """融合 RoPE (前向/反向各 1 个 kernel)。前向与 eager+half 逐位等价。"""

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        _build()
        qo, ko = _EXT.rope_fwd(q, k, cos, sin)
        ctx.save_for_backward(q, k, cos, sin)
        return qo, ko

    @staticmethod
    def backward(ctx, dqo, dko):
        q, k, cos, sin = ctx.saved_tensors
        # 只用到其中一个输出时 autograd 会传 None：补零跑 kernel，但返回 None，
        # 避免给未被使用的分支分配/累加一个全零梯度。
        q_unused, k_unused = dqo is None, dko is None
        if q_unused and k_unused:
            return None, None, None, None
        if q_unused:
            dqo = torch.zeros_like(q)
        elif dqo.stride(3) != 1:
            dqo = dqo.contiguous()
        if k_unused:
            dko = torch.zeros_like(k)
        elif dko.stride(3) != 1:
            dko = dko.contiguous()
        dq, dk = _EXT.rope_bwd(dqo, dko, cos, sin, q, k)
        return (None if q_unused else dq), (None if k_unused else dk), None, None


def rope_can_use(q, k, cos, sin):
    return (_fa_ready(q) and _fa_ready(k)
            and q.shape == k.shape
            and cos.dtype == torch.float32 and sin.dtype == torch.float32
            and cos.device == q.device and sin.device == q.device
            and cos.shape == (q.size(2), 64) and sin.shape == cos.shape
            and cos.stride(1) == 1 and sin.stride(1) == 1)


def rope(q, k, cos2d, sin2d):
    """q,k: (B,H,S,64) fp16 任意步长视图; cos2d/sin2d: (S,64) fp32 (行可 padding)。"""
    return _RopeV100Cuda.apply(q, k, cos2d, sin2d)


def flash_attn_v100(q, k, v, dropout_p: float = 0.0, causal: bool = True,
                    softmax_scale=None):
    if not (q.is_cuda and q.dtype == torch.float16 and q.size(3) == 64):
        raise ValueError('flash_attn_v100 只支持 CUDA fp16 且 head_dim=64 的输入')
    # q/k 通常已经是连续的；v 常常是 fused-qkv 的非连续视图。
    # 非连续本身没关系（kernel 支持任意行跨度），只要跨度 32B 对齐即可。
    if not _fa_ready(q) or not _fa_ready(k) or not _fa_ready(v):
        q, k, v = (t if _fa_ready(t) else t.contiguous() for t in (q, k, v))
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return _FlashAttnV100Cuda.apply(q, k, v, float(dropout_p), bool(causal),
                                    float(softmax_scale))
