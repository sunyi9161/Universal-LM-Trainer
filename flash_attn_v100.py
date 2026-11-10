# -*- coding: utf-8 -*-
"""
flash_attn_v100.py — FlashAttention-2 (forward + backward) for NVIDIA V100 / SM70.

Triton 3.x 在 Volta 上可以正常编译 fp16 `tl.dot`，本文件用标准 FlashAttention-2
分块算法实现，显存 O(N)，不物化 N×N attention 矩阵。

接口（布局与 train.py 内部一致，B,H,S,D）：
    flash_attn_v100(q, k, v, dropout_p=0.0, causal=True, softmax_scale=None)

    q, k, v: (B, H, S, D) fp16，允许非连续（stride 会传进 kernel）
    return : (B, H, S, D) fp16

支持:
    - D in {16, 32, 64, 128}（其他 <=128 的 2 的幂也可以）
    - causal / non-causal
    - attention dropout（Philox，前向反向用同一 mask，不保存 N×N mask）
    - 反向传播（dQ / dK / dV）

注意: V100 没有 cp.async / TMA，num_stages 固定为 1；性能上限低于 A100/H100，
但相对 math 后端仍有数量级提升，并且显存是线性的。
"""

import os

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)

# 全局 seed 计数器（避免每次 forward 都做一次 CPU<->GPU 同步）
_SEED = 1234567


def _next_seed():
    global _SEED
    _SEED = (_SEED * 1103515245 + 12345) & 0x3FFFFFFF
    return _SEED


# ============================================================================
# Forward
# ============================================================================
@triton.jit
def _attn_fwd(
    Q, K, V, O, LSE,
    qsb, qsh, qss, qsd,
    ksb, ksh, kss, ksd,
    vsb, vsh, vss, vsd,
    osb, osh, oss, osd,
    lsb, lsh, lss,
    N,
    scale,
    dropout_p,
    philox_seed,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    CAUSAL: tl.constexpr, DROPOUT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, BD)

    m_mask = offs_m < N
    d_mask = offs_d < BD

    q_base = Q + b * qsb + h * qsh
    k_base = K + b * ksb + h * ksh
    v_base = V + b * vsb + h * vsh

    q = tl.load(q_base + offs_m[:, None] * qss + offs_d[None, :] * qsd,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    m_i = tl.full([BM], -float('inf'), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, BD], tl.float32)

    hi = N if not CAUSAL else tl.minimum(N, (pid_m + 1) * BM)

    for start_n in range(0, hi, BN):
        offs_n_cur = start_n + offs_n
        n_mask = offs_n_cur < N

        k = tl.load(k_base + offs_n_cur[:, None] * kss + offs_d[None, :] * ksd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * (scale * LOG2E)  # log2 单位

        valid = n_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n_cur[None, :])
        qk = tl.where(valid, qk, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, 1)
        p16 = p.to(tl.float16)
        if DROPOUT:
            keep = tl.rand(philox_seed + b * H + h,
                           offs_m[:, None] * N + offs_n_cur[None, :]) >= dropout_p
            p16 = tl.where(keep, p16 * (1.0 / (1.0 - dropout_p)), 0.0).to(tl.float16)
        v = tl.load(v_base + offs_n_cur[:, None] * vss + offs_d[None, :] * vsd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p16, v)
        m_i = m_new

    acc = acc / l_i[:, None]
    lse = (m_i + tl.log2(l_i)) * LN2  # 自然对数 LSE

    o_base = O + b * osb + h * osh
    tl.store(o_base + offs_m[:, None] * oss + offs_d[None, :] * osd,
             acc.to(tl.float16), mask=m_mask[:, None] & d_mask[None, :])
    tl.store(LSE + b * lsb + h * lsh + offs_m * lss, lse, mask=m_mask)


# ============================================================================
# Backward: dK / dV
# ============================================================================
@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, dO, LSE, DELTA, dK, dV,
    qsb, qsh, qss, qsd,
    ksb, ksh, kss, ksd,
    vsb, vsh, vss, vsd,
    dosb, dosh, doss, dosd,
    lsb, lsh, lss,
    dsb, dsh, dss,
    dksb, dksh, dkss, dksd,
    dvsb, dvsh, dvss, dvsd,
    N,
    scale,
    dropout_p,
    philox_seed,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    CAUSAL: tl.constexpr, DROPOUT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, BD)
    n_mask = offs_n < N
    d_mask = offs_d < BD

    k_base = K + b * ksb + h * ksh
    v_base = V + b * vsb + h * vsh
    q_base = Q + b * qsb + h * qsh
    do_base = dO + b * dosb + h * dosh

    k = tl.load(k_base + offs_n[:, None] * kss + offs_d[None, :] * ksd,
                mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    v = tl.load(v_base + offs_n[:, None] * vss + offs_d[None, :] * vsd,
                mask=n_mask[:, None] & d_mask[None, :], other=0.0)

    dk = tl.zeros([BN, BD], tl.float32)
    dv = tl.zeros([BN, BD], tl.float32)

    start_m = 0 if not CAUSAL else (pid_n * BN // BM) * BM
    for start_m_cur in range(start_m, N, BM):
        offs_m = start_m_cur + tl.arange(0, BM)
        m_mask = offs_m < N

        q = tl.load(q_base + offs_m[:, None] * qss + offs_d[None, :] * qsd,
                    mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        do = tl.load(do_base + offs_m[:, None] * doss + offs_d[None, :] * dosd,
                     mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        lse = tl.load(LSE + b * lsb + h * lsh + offs_m * lss, mask=m_mask, other=0.0)
        delta = tl.load(DELTA + b * dsb + h * dsh + offs_m * dss, mask=m_mask, other=0.0)

        s = tl.dot(q, tl.trans(k)) * (scale * LOG2E)
        valid = n_mask[None, :] & m_mask[:, None]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        s = tl.where(valid, s, -float('inf'))

        p = tl.exp2(s - lse[:, None] * LOG2E)
        p = tl.where(valid, p, 0.0)

        if DROPOUT:
            keep = tl.rand(philox_seed + b * H + h,
                           offs_m[:, None] * N + offs_n[None, :]) >= dropout_p
            p = tl.where(keep, p * (1.0 / (1.0 - dropout_p)), 0.0)

        dv += tl.dot(tl.trans(p).to(tl.float16), do)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds).to(tl.float16), q) * scale

    dk_base = dK + b * dksb + h * dksh
    dv_base = dV + b * dvsb + h * dvsh
    tl.store(dk_base + offs_n[:, None] * dkss + offs_d[None, :] * dksd,
             dk.to(tl.float16), mask=n_mask[:, None] & d_mask[None, :])
    tl.store(dv_base + offs_n[:, None] * dvss + offs_d[None, :] * dvsd,
             dv.to(tl.float16), mask=n_mask[:, None] & d_mask[None, :])


# ============================================================================
# Backward: dQ
# ============================================================================
@triton.jit
def _attn_bwd_dq(
    Q, K, V, dO, LSE, DELTA, dQ,
    qsb, qsh, qss, qsd,
    ksb, ksh, kss, ksd,
    vsb, vsh, vss, vsd,
    dosb, dosh, doss, dosd,
    lsb, lsh, lss,
    dsb, dsh, dss,
    dqsb, dqsh, dqss, dqsd,
    N,
    scale,
    dropout_p,
    philox_seed,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    CAUSAL: tl.constexpr, DROPOUT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, BD)
    m_mask = offs_m < N
    d_mask = offs_d < BD

    q_base = Q + b * qsb + h * qsh
    k_base = K + b * ksb + h * ksh
    v_base = V + b * vsb + h * vsh
    do_base = dO + b * dosb + h * dosh

    q = tl.load(q_base + offs_m[:, None] * qss + offs_d[None, :] * qsd,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    do = tl.load(do_base + offs_m[:, None] * doss + offs_d[None, :] * dosd,
                 mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    lse = tl.load(LSE + b * lsb + h * lsh + offs_m * lss, mask=m_mask, other=0.0)
    delta = tl.load(DELTA + b * dsb + h * dsh + offs_m * dss, mask=m_mask, other=0.0)

    dq = tl.zeros([BM, BD], tl.float32)
    hi = N if not CAUSAL else tl.minimum(N, (pid_m + 1) * BM)

    for start_n in range(0, hi, BN):
        offs_n_cur = start_n + offs_n
        n_mask = offs_n_cur < N

        k = tl.load(k_base + offs_n_cur[:, None] * kss + offs_d[None, :] * ksd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_base + offs_n_cur[:, None] * vss + offs_d[None, :] * vsd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        s = tl.dot(q, tl.trans(k)) * (scale * LOG2E)
        valid = n_mask[None, :] & m_mask[:, None]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n_cur[None, :])
        s = tl.where(valid, s, -float('inf'))

        p = tl.exp2(s - lse[:, None] * LOG2E)
        p = tl.where(valid, p, 0.0)

        if DROPOUT:
            keep = tl.rand(philox_seed + b * H + h,
                           offs_m[:, None] * N + offs_n_cur[None, :]) >= dropout_p
            p = tl.where(keep, p * (1.0 / (1.0 - dropout_p)), 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.float16), k) * scale

    dq_base = dQ + b * dqsb + h * dqsh
    tl.store(dq_base + offs_m[:, None] * dqss + offs_d[None, :] * dqsd,
             dq.to(tl.float16), mask=m_mask[:, None] & d_mask[None, :])


# ============================================================================
# autograd wrapper
def _pick_blocks(D):
    """V100 只有 96KB shared memory，按 head_dim 选 block。

    head_dim=64 (本模型) 用 64x64 最快；128 时寄存器/shared 放不下，退到 32x32。
    可用环境变量 FA_V100_BM / FA_V100_BN 覆盖（调参用）。
    """
    bm = int(os.environ.get('FA_V100_BM', '0'))
    bn = int(os.environ.get('FA_V100_BN', '0'))
    if bm and bn:
        return bm, bn
    if D >= 128:
        return 32, 32
    return 64, 64


def _pick_tuning(D):
    """返回 (num_warps, num_stages)。V100 无 cp.async/TMA，stage 多了没用还可能放不下。"""
    bm, bn = _pick_blocks(D)
    warps = int(os.environ.get('FA_V100_WARPS', '0')) or (4 if bm * bn <= 4096 else 8)
    stages = int(os.environ.get('FA_V100_STAGES', '0')) or 1
    return warps, stages


# ============================================================================
class _FlashAttnV100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, causal, softmax_scale):
        B, H, N, D = q.shape
        assert D in (16, 32, 64, 128), f'unsupported head_dim={D}'
        assert q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16

        o = torch.empty((B, H, N, D), device=q.device, dtype=q.dtype)
        lse = torch.zeros((B, H, N), device=q.device, dtype=torch.float32)
        BD = triton.next_power_of_2(D)
        BM, BN = _pick_blocks(D)
        warps, stages = _pick_tuning(D)
        seed = _next_seed()
        grid = (triton.cdiv(N, BM), B * H)
        _attn_fwd[grid](
            q, k, v, o, lse,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            lse.stride(0), lse.stride(1), lse.stride(2),
            N, softmax_scale, dropout_p, seed,
            H, BM, BN, BD,
            causal, dropout_p > 0.0,
            num_warps=warps, num_stages=stages,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.dropout_p = dropout_p
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.seed = seed
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, N, D = q.shape
        do = do.contiguous()
        delta = (do.float() * o.float()).sum(-1).contiguous().float()

        dq = torch.empty((B, H, N, D), device=q.device, dtype=q.dtype)
        dk = torch.empty((B, H, N, D), device=q.device, dtype=q.dtype)
        dv = torch.empty((B, H, N, D), device=q.device, dtype=q.dtype)

        BD = triton.next_power_of_2(D)
        BM, BN = _pick_blocks(D)
        warps, stages = _pick_tuning(D)
        dropout_p = ctx.dropout_p
        causal = ctx.causal
        scale = ctx.softmax_scale

        grid_n = (triton.cdiv(N, BN), B * H)
        _attn_bwd_dkdv[grid_n](
            q, k, v, do, lse, delta, dk, dv,
            *q.stride(), *k.stride(), *v.stride(),
            *do.stride(), lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            *dk.stride(), *dv.stride(),
            N, scale, dropout_p, ctx.seed,
            H, BM, BN, BD,
            causal, dropout_p > 0.0,
            num_warps=warps, num_stages=stages,
        )

        grid_m = (triton.cdiv(N, BM), B * H)
        _attn_bwd_dq[grid_m](
            q, k, v, do, lse, delta, dq,
            *q.stride(), *k.stride(), *v.stride(),
            *do.stride(), lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            *dq.stride(),
            N, scale, dropout_p, ctx.seed,
            H, BM, BN, BD,
            causal, dropout_p > 0.0,
            num_warps=warps, num_stages=stages,
        )
        return dq, dk, dv, None, None, None


def flash_attn_v100(q, k, v, dropout_p: float = 0.0, causal: bool = True,
                    softmax_scale=None):
    """q,k,v: (B,H,S,D) fp16 -> o: (B,H,S,D) fp16"""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return _FlashAttnV100.apply(q, k, v, float(dropout_p), bool(causal), float(softmax_scale))
