#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 Decoder-only 语言模型训练脚本（CUDA，V100 优化）

通过 config.json（或 --config 指定）配置模型结构、训练超参、分词器和路径，
同一份代码可用于训练从小到大的各种 Decoder-only 模型。

主要能力:
1. 配置驱动
   - d_model / n_heads / n_layers / d_ff / max_seq_len / dropout / tie_weights
   - batch_size / grad_accum / lr / warmup / token budget / epochs
   - data_dir / cache_dir / checkpoint_dir / tokenizer 路径
2. 数据管线
   - 一次性把分桶 pkl 或 JSONL 转成 uint16 平铺缓存 + offsets
   - np.memmap 零拷贝随机访问，单样本读取约 2KB
   - 按长度分桶，桶边界由 max_seq_len 自动生成（步长 50）
3. 变长微批 (Token-Budget Micro Batching)
   - 每个微批约 target_tokens 个 token，短序列不空转
   - 每个优化器 step 严格消费 batch_size × grad_accum 个样本
4. 分块 cross-entropy
   - 不物化 [B*S, vocab] 的 fp32 logits
5. fused AdamW + 去掉每步 cuda.synchronize()/loss.item()
6. F.rms_norm 融合 RMSNorm（数学等价）
7. torch.compile 只编译 forward_hidden；UNIVERSAL_LM_COMPILE=0 或
   UNIVERSAL_LM_COMPILE=0（兼容 EMINI_COMPILE=0）可关闭
8. 可恢复训练（--resume）
   - Ctrl+C 菜单显式保存 <checkpoint_dir>/last.pt：模型 / 优化器 / GradScaler /
     调度器 / RNG / 已消费样本位置
   - 菜单 7 保存普通检查点 checkpoint_epoch{E}_step{S}.pt（不可续训）
   - 只在优化器 step 边界落盘，恢复时按已消费样本数重放采样器
   - 数据指纹（tokenizer + 分桶计数 + BOS/EOS 开关）不匹配时拒绝恢复
9. BOS/EOS 监督（默认开启）
   - 每个样本编码为 [BOS] + 正文 + [EOS]，超长截断正文
   - --no-bos / --no-eos 可关闭，开关写入数据指纹
   - 文本里字面出现的 "<bos>"/"<eos>" 按普通 BPE 编码
10. FlashAttention-2
   - csrc/flash_attn_v100 提供 sm70 CUDA 前向/反向；不可用时回退 SDPA
"""

import argparse
import json
import array
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Dict, Tuple
import os
import signal
import sys
from tqdm import tqdm
import warnings
import glob
import random
from dataclasses import dataclass, asdict
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers.trainers import BpeTrainer
import time
from datetime import datetime, timedelta
import threading
import psutil
import pickle
import numpy as np
import hashlib
import shutil
from collections import deque
import atexit
from functools import lru_cache, partial

warnings.filterwarnings('ignore')

# ==================== FlashAttention-2 (V100/sm70) ====================
# 手写的 sm70 CUDA FA2（前向 + 反向）。官方 flash-attn 不支持 Volta，
# Triton 在 Volta 上会退化成 FMA，所以在 V100 上只有这条路能真正用上张量核心。
# 用 FA_V100_DISABLE=1 可强制回退到 SDPA（mem_efficient）。
try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if _THIS_DIR not in sys.path:
        sys.path.append(_THIS_DIR)
    import flash_attn_v100_cuda as _fa_v100
except Exception as _e:  # pragma: no cover
    _fa_v100 = None
    print(f'[flash_attn_v100] 未启用: {_e}')

_FA_V100_ENABLED = (os.environ.get('FA_V100_DISABLE', '0') != '1')
if _FA_V100_ENABLED and _fa_v100 is not None:
    # 第一次会现场编译 (约 1 分钟), 之后复用 build 目录
    if _fa_v100.is_available():
        print('[flash_attn_v100] V100(sm70) 手写 FlashAttention-2 已启用')
    else:
        print(f'[flash_attn_v100] 编译失败, 回退 SDPA: '
              f'{str(_fa_v100.build_error())[-500:]}')
        _fa_v100 = None


def _fa_v100_usable(q, k, v):
    return (_FA_V100_ENABLED and _fa_v100 is not None
            and _fa_v100.can_use(q, k, v))


def _fa_v100_half_inputs(q, k, v):
    """autocast 下 SDPA 的语义本来就是"把 fp32 输入降成 fp16 再算"。

    注意 RoPE 里 cos/sin 是 fp32 缓存，会把 q/k promote 回 fp32（v 仍是 fp16），
    所以这里显式做一次和 autocast 完全一致的降精度，否则 kernel 用不上。
    非 autocast 场景不动 dtype，由 can_use 决定是否回退。
    """
    if not torch.is_autocast_enabled():
        return q, k, v
    return tuple(t.half() if t.dtype == torch.float32 else t
                 for t in (q, k, v))

# ==================== 线程安全的全局变量 ====================
_interrupted = False
_interrupt_lock = threading.Lock()
_menu_shown = False
_current_model = None
_current_optimizer = None
_current_scaler = None
_current_scheduler = None
_current_epoch = None
_current_step = None
_save_dir = './checkpoints'
_TOTAL_SAMPLES_PER_STEP = 64   # batch_size(4) * grad_accum(16)
SAMPLER_VERSION = 1            # 采样算法版本：改变采样逻辑时必须 +1，旧 checkpoint 将拒绝恢复

# ---- 可恢复训练状态 ----
# _TRAIN_STATE 只在“一个优化器 step 完全结束”时更新，因此任何时刻落盘的都是
# 一个干净的 step 边界：优化器状态与已消费样本数严格对齐，恢复后不重不漏。
CHECKPOINT_FORMAT_VERSION = 2
_TRAIN_STATE = None
_current_save_path = './checkpoints/last.pt'

test_prompts = [
    "今天",
    "中国的首都是",
    "地球围绕",
    "1+1=",
    "在一个漆黑的夜晚",
    "import pygame",
    "There is a"
]

# ==================== 辅助函数 ====================
def get_original_model(model):
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    elif hasattr(model, '_forward_module') and hasattr(model._forward_module, '_orig_mod'):
        return model._forward_module._orig_mod
    return model

def get_memory_usage():
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return mem_gb

def _capture_rng_state():
    """保存所有会影响训练随机性的 RNG 状态（dropout / 数据顺序）。"""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state['cuda'] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def _restore_rng_state(state):
    if not state:
        return
    try:
        if 'python' in state:
            random.setstate(state['python'])
        if 'numpy' in state:
            np.random.set_state(state['numpy'])
        if 'torch' in state:
            torch.set_rng_state(state['torch'])
        if 'cuda' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda'])
        print("✓ RNG 状态已恢复")
    except Exception as e:
        print(f"⚠️ RNG 状态恢复失败（不影响数据拼接）: {e}")


def save_checkpoint(path: Optional[str] = None, quiet: bool = False):
    """保存**完整**训练进度（模型/优化器/调度器/scaler/RNG/已消费样本位置）。

    采用临时文件 + os.replace 原子替换：即使保存到一半断电/被杀，旧的
    last.pt 仍然完整可用。
    """
    global _current_model, _current_optimizer, _current_scaler, _current_scheduler
    global _TRAIN_STATE

    if _current_model is None:
        if not quiet:
            print("模型不存在，无法保存")
        return False
    if _TRAIN_STATE is None:
        if not quiet:
            print("还没有完成任何优化器 step，暂无可保存的训练进度")
        return False

    path = path or _current_save_path
    try:
        model_to_save = get_original_model(_current_model)
        save_dict = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            # 以普通 dict 保存模型结构参数：inf.py 可直接加载本文件（无需 --export）。
            'config': asdict(model_to_save.config),
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': _current_optimizer.state_dict() if _current_optimizer else None,
            'scaler_state_dict': _current_scaler.state_dict() if _current_scaler is not None else None,
            'scheduler_state_dict': {
                'current_step': _current_scheduler.current_step,
                'warmup_steps': _current_scheduler.warmup_steps,
                'total_steps': _current_scheduler.total_steps,
                'base_lr': _current_scheduler.base_lr,
                'min_lr': _current_scheduler.min_lr,
            } if _current_scheduler is not None else None,
            'rng_state': _TRAIN_STATE.get('rng_state') or _capture_rng_state(),
            'training': dict(_TRAIN_STATE),
            # 供 inf.py 快速判断生成/评分是否按“带 BOS/EOS”口径处理。
            'eos_supervised': bool(
                (_TRAIN_STATE.get('data_fingerprint') or {}).get('append_eos', False)),
            'eos_id': (_TRAIN_STATE.get('data_fingerprint') or {}).get('eos_id'),
            'bos_supervised': bool(
                (_TRAIN_STATE.get('data_fingerprint') or {}).get('prepend_bos', False)),
            'bos_id': (_TRAIN_STATE.get('data_fingerprint') or {}).get('bos_id'),
        }

        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            torch.save(save_dict, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)          # 原子替换

        # 轻量元数据 sidecar：启动时可先快速校验 tokenizer/cache，避免白建 21GB 缓存
        try:
            meta = {
                'format_version': CHECKPOINT_FORMAT_VERSION,
                'data_fingerprint': _TRAIN_STATE.get('data_fingerprint'),
                'epoch': _TRAIN_STATE.get('epoch'),
                'step_in_epoch': _TRAIN_STATE.get('step_in_epoch'),
                'global_step': _TRAIN_STATE.get('global_step'),
                'epoch_samples': _TRAIN_STATE.get('epoch_samples'),
                'saved_at': _TRAIN_STATE.get('saved_at'),
                'eos_supervised': bool(
                    (_TRAIN_STATE.get('data_fingerprint') or {}).get('append_eos', False)),
                'eos_id': (_TRAIN_STATE.get('data_fingerprint') or {}).get('eos_id'),
                'bos_supervised': bool(
                    (_TRAIN_STATE.get('data_fingerprint') or {}).get('prepend_bos', False)),
                'bos_id': (_TRAIN_STATE.get('data_fingerprint') or {}).get('bos_id'),
            }
            meta_tmp = path + '.json.tmp'
            with open(meta_tmp, 'w') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(meta_tmp, path + '.json')
        except Exception as e:
            print(f"⚠️ 写入进度元数据失败（不影响 checkpoint）: {e}")

        if not quiet:
            size_gb = os.path.getsize(path) / 1e9
            tr = _TRAIN_STATE
            print(f"\n✓ 训练进度已保存: {path} ({size_gb:.2f} GB) | "
                  f"epoch={tr.get('epoch')} global_step={tr.get('global_step')} "
                  f"epoch_step={tr.get('step_in_epoch')} "
                  f"已消费={tr.get('epoch_samples'):,} 样本")
        return True
    except Exception as e:
        print(f"\n✗ 保存训练进度失败: {e}")
        try:
            os.remove(path + '.tmp')
        except OSError:
            pass
        return False


def save_plain_checkpoint(quiet: bool = False):
    """保存**普通检查点**（与旧版行为一致）：模型 / 优化器 / scaler / 调度器 / config。

    文件名为 checkpoints/checkpoint_epoch{E}_step{S}.pt，可被 inf.py 直接加载，
    也可用于另起训练分支；但它不含“已消费样本位置”和 RNG，**不能**用 --resume
    严格续训（想续训请用菜单 1/3/6 写入的 last.pt）。
    """
    global _current_model, _current_optimizer, _current_scaler, _current_scheduler
    global _TRAIN_STATE

    if _current_model is None:
        if not quiet:
            print("模型不存在，无法保存")
        return None

    tr = _TRAIN_STATE or {}
    epoch = int(tr.get('epoch', 0))
    step = int(tr.get('global_step', 0))
    path = os.path.join(_save_dir, f'checkpoint_epoch{epoch}_step{step}.pt')
    try:
        model_to_save = get_original_model(_current_model)
        save_dict = {
            'checkpoint_kind': 'plain',
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'config': asdict(model_to_save.config),
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': _current_optimizer.state_dict() if _current_optimizer else None,
            'scaler_state_dict': _current_scaler.state_dict() if _current_scaler is not None else None,
            'scheduler_state_dict': {
                'current_step': _current_scheduler.current_step,
                'warmup_steps': _current_scheduler.warmup_steps,
                'total_steps': _current_scheduler.total_steps,
                'base_lr': _current_scheduler.base_lr,
                'min_lr': _current_scheduler.min_lr,
            } if _current_scheduler is not None else None,
            'epoch': epoch,
            'step_in_epoch': int(tr.get('step_in_epoch', 0)),
            'global_step': step,
            'eos_supervised': bool(
                (tr.get('data_fingerprint') or {}).get('append_eos', False)),
            'eos_id': (tr.get('data_fingerprint') or {}).get('eos_id'),
            'bos_supervised': bool(
                (tr.get('data_fingerprint') or {}).get('prepend_bos', False)),
            'bos_id': (tr.get('data_fingerprint') or {}).get('bos_id'),
        }

        os.makedirs(_save_dir, exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            torch.save(save_dict, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)          # 原子替换

        if not quiet:
            size_gb = os.path.getsize(path) / 1e9
            print(f"\n✓ 普通检查点已保存: {path} ({size_gb:.2f} GB) | "
                  f"epoch={epoch} global_step={step}")
        return path
    except Exception as e:
        print(f"\n✗ 保存普通检查点失败: {e}")
        try:
            os.remove(path + '.tmp')
        except OSError:
            pass
        return None


# 旧版 tokenizer.json 把 <bos>/<eos> 作为 added token（文本里字面出现时会被
# 匹配成 id 2/3）。现在文件里已移除这两项，但缓存/checkpoint 的 tokenizer_fp
# 是按“含这两项”的序列化结果算的。指纹函数在计算前把缺失的两项按原样补回去，
# 这样新旧分词器得到完全相同的指纹，--resume 不会被无意义地拒绝。
_CANONICAL_ADDED = (
    {'id': 2, 'content': '<bos>', 'single_word': False, 'lstrip': False,
     'rstrip': False, 'normalized': False, 'special': True},
    {'id': 3, 'content': '<eos>', 'single_word': False, 'lstrip': False,
     'rstrip': False, 'normalized': False, 'special': True},
)


def _canonical_tokenizer_str(core: Tokenizer) -> str:
    cfg = json.loads(core.to_str())
    added = cfg.get('added_tokens', [])
    have = {t.get('content') for t in added}
    vocab = cfg.get('model', {}).get('vocab', {})
    extra = [dict(t) for t in _CANONICAL_ADDED
             if t['content'] not in have and vocab.get(t['content']) == t['id']]
    if not extra:
        return core.to_str()
    cfg['added_tokens'] = sorted(added + extra, key=lambda t: t['id'])
    return Tokenizer.from_str(json.dumps(cfg, ensure_ascii=False)).to_str()


def tokenizer_fingerprint(tokenizer) -> Optional[str]:
    """分词器内容的指纹（与 FlatBucketCache._tokenizer_fingerprint 保持一致）。"""
    try:
        core = getattr(tokenizer, 'tokenizer', tokenizer)
        return hashlib.md5(
            _canonical_tokenizer_str(core).encode('utf-8')).hexdigest()[:16]
    except Exception:
        return None


def strip_bos_eos_added(core: Tokenizer) -> Tokenizer:
    """返回一个去掉 <bos>/<eos> added token 的副本（用于编码和保存）。

    这两项仍保留在 model.vocab 的 id 2/3 上，所以 token_to_id/bos_token_id
    不受影响；去掉后文本里字面出现的 "<bos>"/"<eos>" 会按普通 BPE 处理。
    """
    cfg = json.loads(core.to_str())
    added = cfg.get('added_tokens', [])
    plain = [t for t in added if t.get('content') not in ('<bos>', '<eos>')]
    if len(plain) == len(added):
        return core
    cfg['added_tokens'] = plain
    return Tokenizer.from_str(json.dumps(cfg, ensure_ascii=False))


def _diff_fingerprints(saved: dict, current: dict) -> str:
    keys = sorted(set(saved) | set(current))
    lines = []
    for k in keys:
        a, b = saved.get(k), current.get(k)
        if a != b:
            if isinstance(a, (list, str)) and len(str(a)) > 120:
                a = f"{str(a)[:117]}..."
            if isinstance(b, (list, str)) and len(str(b)) > 120:
                b = f"{str(b)[:117]}..."
            lines.append(f"    {k}: checkpoint={a!r}  当前={b!r}")
    return "\n".join(lines)


def load_training_checkpoint(path, model, optimizer, scaler, scheduler,
                             data_fingerprint, device):
    """加载完整训练进度。返回恢复位置；数据不一致时直接抛错，绝不静默错位。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到训练进度文件: {path}")

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    fmt = ckpt.get('format_version')
    if fmt != CHECKPOINT_FORMAT_VERSION:
        print(f"⚠️ checkpoint 格式版本 {fmt} != {CHECKPOINT_FORMAT_VERSION}，仍尝试加载")

    training = ckpt.get('training')
    if not training:
        raise RuntimeError(
            f"{path} 是普通检查点（只含模型/优化器，不含样本位置与 RNG），不能用于 --resume 续训。\n"
            "请改用菜单 1/3/6 保存的 last.pt；"
            "若只想接着这份权重训练，去掉 --resume 并用普通方式加载该权重。"
        )
    saved_fp = training.get('data_fingerprint')
    if saved_fp is not None:
        # 兼容旧 checkpoint：指纹里后来新增的字段（如 target_tokens/max_micro_batch）
        # 在旧文件里不存在。缺失字段无法事后验证，只能按当前值继续，但必须显式警告；
        # 已存在的字段仍严格比较，不一致绝不恢复。
        missing = sorted(set(data_fingerprint) - set(saved_fp))
        mismatched = {k: (saved_fp[k], data_fingerprint[k])
                      for k in data_fingerprint
                      if k in saved_fp and saved_fp[k] != data_fingerprint[k]}
        if mismatched:
            if set(mismatched) <= {'append_eos', 'eos_id', 'prepend_bos', 'bos_id'}:
                raise RuntimeError(
                    "训练进度与当前的 BOS/EOS 设置不一致，拒绝恢复"
                    "（否则同一份 checkpoint 会静默改用不同的训练目标）。\n"
                    "差异:\n" + _diff_fingerprints(
                        {k: v[0] for k, v in mismatched.items()},
                        {k: v[1] for k, v in mismatched.items()}) +
                    "\n如果想让本次训练沿用 checkpoint 的设置，请检查是否多加了 "
                    "--no-bos/--no-eos。"
                )
            raise RuntimeError(
                "训练进度与当前数据不匹配，拒绝恢复（否则样本 id 会错位，必然重复/遗漏）。\n"
                "差异:\n" + _diff_fingerprints(
                    {k: v[0] for k, v in mismatched.items()},
                    {k: v[1] for k, v in mismatched.items()}) +
                "\n如果确实换了 tokenizer/缓存/分桶，请从头训练（不要加 --resume）。"
            )
        if missing:
            print("⚠️ 旧 checkpoint 的数据指纹缺少以下字段（保存时本脚本还没有校验它们）:")
            for k in missing:
                print(f"    {k}: 当前={data_fingerprint[k]!r}")
            print("   将按当前值继续。请确认这些参数自 checkpoint 保存以来没有改动过，"
                  "否则样本顺序可能错位。")
            if 'append_eos' in missing or 'prepend_bos' in missing:
                print("   ⚠️ 该 checkpoint 保存时还没有 BOS/EOS 监督；本次将按当前设置"
                      f"（BOS={data_fingerprint.get('prepend_bos')}, "
                      f"EOS={data_fingerprint.get('append_eos')}）继续。"
                      "loss 会因新增 target 而变化，但样本顺序不变，不会重复/遗漏。")


    model_to_load = get_original_model(model)
    model_to_load.load_state_dict(ckpt['model_state_dict'])

    if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler is not None and ckpt.get('scaler_state_dict') is not None:
        try:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        except Exception as e:
            print(f"⚠️ scaler 状态恢复失败: {e}")

    sched = ckpt.get('scheduler_state_dict')
    if scheduler is not None and sched is not None:
        if int(sched.get('total_steps', scheduler.total_steps)) != scheduler.total_steps:
            print(f"⚠️ 调度器 total_steps 不一致: checkpoint={sched.get('total_steps')} "
                  f"当前配置={scheduler.total_steps}（将沿用 checkpoint 的原 schedule）")
        scheduler.current_step = int(sched.get('current_step', 0))
        scheduler.warmup_steps = int(sched.get('warmup_steps', scheduler.warmup_steps))
        scheduler.total_steps = int(sched.get('total_steps', scheduler.total_steps))
        scheduler.base_lr = float(sched.get('base_lr', scheduler.base_lr))
        scheduler.min_lr = float(sched.get('min_lr', scheduler.min_lr))
        # 立刻把 lr 对齐到 current_step
        for pg in optimizer.param_groups:
            if scheduler.current_step <= scheduler.warmup_steps and scheduler.warmup_steps > 0:
                pg['lr'] = scheduler.base_lr * (scheduler.current_step / scheduler.warmup_steps)
            else:
                progress = min(1.0, max(0, scheduler.current_step - scheduler.warmup_steps) /
                               max(1, scheduler.total_steps - scheduler.warmup_steps))
                pg['lr'] = scheduler.min_lr + (scheduler.base_lr - scheduler.min_lr) * \
                           (1 + math.cos(math.pi * progress)) / 2

    _restore_rng_state(ckpt.get('rng_state'))

    info = {
        'epoch': int(training.get('epoch', 1)),
        'step_in_epoch': int(training.get('step_in_epoch', 0)),
        'global_step': int(training.get('global_step', 0)),
        'epoch_samples': int(training.get('epoch_samples', 0)),
        'epoch_loss_sum': float(training.get('epoch_loss_sum', 0.0)),
        'best_loss': float(training.get('best_loss', float('inf'))),
        'path': path,
    }
    print(f"\n✓ 已恢复训练进度: {path}")
    print(f"  - epoch={info['epoch']}  epoch_step={info['step_in_epoch']}  "
          f"global_step={info['global_step']}")
    print(f"  - 本 epoch 已消费样本: {info['epoch_samples']:,}")
    return info


def _snapshot_train_state(epoch, step_in_epoch, global_step, epoch_samples,
                          epoch_loss_sum, best_loss, data_fingerprint):
    """只在优化器 step 边界调用：保证落盘状态与样本消费位置严格对齐。"""
    global _TRAIN_STATE
    _TRAIN_STATE = {
        'epoch': int(epoch),
        'step_in_epoch': int(step_in_epoch),
        'global_step': int(global_step),
        'epoch_samples': int(epoch_samples),
        'epoch_loss_sum': float(epoch_loss_sum),
        'best_loss': float(best_loss),
        'data_fingerprint': data_fingerprint,
        'saved_at': time.time(),
        # 关键：RNG 也必须取“step 边界”的快照。否则用户在 step 中途按 Ctrl+C
        # 保存时，会把已经多抽的 dropout 随机数带进去，恢复后前向不再等价。
        'rng_state': _capture_rng_state(),
    }

def generate_test():
    global _current_model
    if _current_model is None:
        print("模型不存在，无法生成")
        return
    print("\n" + "="*60)
    print("代码生成测试")
    print("="*60)
    model_to_gen = get_original_model(_current_model)
    try:
        model_to_gen.eval()
        for prompt in test_prompts:
            output = generate_text(model_to_gen, prompt, max_new_tokens=80)
            print(f"\n提示词: {prompt}")
            print(f"生成: {output}")
            print("-"*50)
    except Exception as e:
        print(f"生成失败: {e}")
    finally:
        model_to_gen.train()

def generate_text(model, prompt: str, max_new_tokens: int = 150, temperature: float = 0.8, top_k: int = 50):
    # 训练时每个样本都以 BOS 开头（prepend_bos=True）；若本次训练用 --no-bos，
    # 则模型属性 prepend_bos=False，这里也不加，避免喂未训练过的 BOS。
    add_bos = bool(getattr(model, 'prepend_bos', True))
    prompt_ids = list(model.tokenizer.encode(prompt))
    input_ids = ([model.tokenizer.bos_token_id] if add_bos else []) + prompt_ids
    if not input_ids:
        input_ids = [model.tokenizer.bos_token_id]
    if len(input_ids) > model.config.max_seq_len:
        input_ids = input_ids[:model.config.max_seq_len]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.shape[1] > model.config.max_seq_len:
                input_tensor = input_tensor[:, -model.config.max_seq_len:]
            logits = model.forward(input_tensor)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                top_logits, top_indices = torch.topk(next_logits, top_k_val)
                mask = torch.full_like(next_logits, float('-inf'))
                mask.scatter_(-1, top_indices, top_logits)
                next_logits = mask
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            if (next_token == model.tokenizer.eos_token_id).all():
                break
    output_text = model.tokenizer.decode(input_tensor[0].tolist())
    return output_text

# ==================== 线程安全的中断处理 ====================
def handle_interrupt(sig, frame):
    global _interrupted
    with _interrupt_lock:
        if not _interrupted:
            _interrupted = True
            print("\n\n" + "="*60)
            print("⚠️ 检测到 Ctrl+C，正在安全停止...")
            print("="*60)

def show_interrupt_menu():
    global _interrupted, _menu_shown

    with _interrupt_lock:
        if not _interrupted or _menu_shown:
            return None
        _menu_shown = True

    print("\n" + "="*60)
    print("训练已暂停，请选择操作:")
    print("="*60)
    print("  1. 💾 保存训练进度（可 --resume）并继续训练")
    print("  2. 🧪 运行生成测试")
    print("  3. 💾 保存训练进度（可 --resume）并退出")
    print("  4. ❌ 退出程序 (不保存)")
    print("  5. ▶️  继续训练 (不保存)")
    print("  6. 💾 只保存训练进度（可恢复），返回菜单")
    print("  7. 💾 保存普通检查点（模型+优化器，可推理/另起分支，不可续训）")
    print("="*60)

    while True:
        try:
            choice = input("请输入选项 (1/2/3/4/5/6/7): ").strip()

            if choice == '1':
                print("\n正在保存训练进度...")
                save_checkpoint()
                print("✓ 继续训练...")
                with _interrupt_lock:
                    _interrupted = False
                    _menu_shown = False
                return 'continue'

            elif choice == '2':
                generate_test()
                with _interrupt_lock:
                    _interrupted = False
                    _menu_shown = False
                return 'continue'

            elif choice == '3':
                save_checkpoint()
                print("✓ 程序正常退出")
                sys.exit(0)

            elif choice == '4':
                print("退出程序 (不保存)")
                sys.exit(0)

            elif choice == '5':
                with _interrupt_lock:
                    _interrupted = False
                    _menu_shown = False
                return 'continue'

            elif choice == '6':
                print("\n正在保存训练进度...")
                if save_checkpoint():
                    print("✓ 已保存，返回菜单（训练仍暂停）")
                else:
                    print("⚠️ 保存失败（可能还没完成第一个 step）")
                continue

            elif choice == '7':
                print("\n正在保存普通检查点...")
                if save_plain_checkpoint():
                    print("✓ 已保存，返回菜单（训练仍暂停）")
                else:
                    print("⚠️ 保存失败")
                continue

            else:
                print("❌ 无效选项，请输入 1, 2, 3, 4, 5, 6 或 7")

        except KeyboardInterrupt:
            print("\n\n⚠️ 请使用菜单选项退出")
            continue
        except EOFError:
            print("\n\n⚠️ 检测到输入错误，默认选择继续训练")
            with _interrupt_lock:
                _interrupted = False
                _menu_shown = False
            return 'continue'

def check_interrupt():
    global _interrupted
    with _interrupt_lock:
        return _interrupted

signal.signal(signal.SIGINT, handle_interrupt)

# ==================== CUDA配置 (V100优化) ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high')

    major, minor = torch.cuda.get_device_capability()
    supports_bf16 = (major, minor) >= (8, 0)
    supports_fp16 = True
    supports_compile = hasattr(torch, 'compile')

    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - Compute Capability: {major}.{minor}")
    print(f"  - Tensor Cores: {'支持' if major >= 7 else '不支持'} (FP16)")
else:
    supports_fp16 = False
    supports_bf16 = False
    supports_compile = False
    print("CUDA not available, using CPU")
print(f"Device: {device}")
print()

# ==================== 配置类 ====================
@dataclass
class ModelConfig:
    vocab_size: int = 66536
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 4096
    dropout: float = 0.1
    use_rope: bool = True
    tie_weights: bool = True
    use_fp16: bool = True
    use_compile: bool = True
    use_gradient_checkpointing: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 6
    max_grad_norm: float = 1.0
    save_every: int = 1
    num_workers: int = 4
    seed: int = 42
    prefetch_factor: int = 4
    log_every: int = 50
    use_fp16: bool = True
    use_compile: bool = True
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    test_mode: bool = False
    test_samples: int = 100000
    # --- 性能相关 ---
    target_tokens: int = 12288      # 每个微批的 token 预算 (显存安全位)
    max_micro_batch: int = 64       # 微批样本数上限 = 每个优化器step的样本数
    fused_optimizer: bool = True
    ce_chunk_rows: int = 2048       # 分块 cross-entropy 的行数
    max_steps: int = 0              # >0 时只跑指定 step 数（调试/基准）
    # --- EOS/BOS 监督 ---
    # 在每个训练样本（= 一篇完整文档）前后加 <bos>/<eos>：EOS 让模型学会
    # “文档结束”，BOS 让模型学会“文档开始”（否则这两行 embedding 永不更新）。
    append_eos: bool = True
    prepend_bos: bool = True
    # --- 进度保存 / 恢复 ---
    resume_path: Optional[str] = None   # 非 None 时从该 checkpoint 恢复

SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']

# 融合 RMSNorm：F.rms_norm 在 CUDA 上是单 kernel，比手动 pow/mean/rsqrt/mul 快 ~3x；
# 数学等价（内部 fp32 归约，比原 fp16 平方更精确）。需要逐位复现旧行为可置 False。
USE_FUSED_RMSNORM = True

# ==================== 分词器训练功能 ====================
def train_tokenizer(data_dir: str, vocab_size: int = 66536, tokenizer_path: str = 'tokenizer.json',
                   min_frequency: int = 2, force_retrain: bool = False,
                   max_bytes: int = 1_200_000_000, max_texts: int = 2_000_000):
    """训练BPE分词器。

    语料按**文件大小比例**分配字节预算 (max_bytes)，每个文件只读属于自己的
    那一份，因此语料构成 = 磁盘上的数据构成，与 glob 顺序无关。
    max_texts 只是单文件条数安全阀。
    """
    print("="*70)
    print(f"BPE分词器训练 (按文件大小分配 {max_bytes/1e6:,.0f} MB 预算)")
    print("="*70)
    if max_bytes > 250_000_000:
        est_gb = 1.5 + max_bytes / 1e6 * 0.035
        print(f"⚠️ 预算 {max_bytes/1e6:,.0f} MB 预计峰值内存 ~{est_gb:.1f} GB；"
              f"实测 300 MB 就会触发 swap 并慢到 25 分钟，建议 <= 200 MB")

    if not force_retrain and os.path.exists(tokenizer_path):
        try:
            temp_tokenizer = Tokenizer.from_file(tokenizer_path)
            existing_vocab = temp_tokenizer.get_vocab_size()
            print(f"✓ 发现已训练的分词器: {tokenizer_path}")
            print(f"  - 词表大小: {existing_vocab}")

            special_tokens_exist = True
            for token in SPECIAL_TOKENS:
                if temp_tokenizer.token_to_id(token) is None:
                    special_tokens_exist = False
                    break

            if special_tokens_exist:
                print("✓ 特殊token完整，无需重新训练")
                return True
            else:
                print("  ⚠️ 特殊token不完整，重新训练...")
        except Exception as e:
            print(f"  ⚠️ 读取tokenizer失败: {e}")

    print(f"\n正在扫描数据目录: {data_dir}")
    jsonl_files = glob.glob(os.path.join(data_dir, '*.jsonl'))
    jsonl_files = [f for f in jsonl_files if 'wt.jsonl' not in f and 'bucketed_cache' not in f]
    jsonl_files.sort()  # 固定顺序，保证每次训练结果可复现

    if not jsonl_files:
        print(f"❌ 在 {data_dir} 中未找到JSONL文件")
        return False

    print(f"找到 {len(jsonl_files)} 个JSONL文件")

    sizes = {f: os.path.getsize(f) for f in jsonl_files}
    total_size = sum(sizes.values()) or 1
    quotas = {f: int(max_bytes * sizes[f] / total_size) for f in jsonl_files}

    print(f"\n按文件大小分配预算 (总计 {max_bytes/1e6:,.0f} MB):")
    for f in jsonl_files:
        print(f"  {os.path.basename(f):<22} {sizes[f]/1e9:7.2f} GB  "
              f"{sizes[f]/total_size:6.2%}  ->  {quotas[f]/1e6:8.1f} MB")

    file_stats = []

    def text_generator():
        for file_path in jsonl_files:
            quota = quotas[file_path]
            used = count = 0
            try:
                f = open(file_path, 'rb')
            except Exception as e:
                print(f"  ⚠️ 打不开 {file_path}: {e}")
                file_stats.append((os.path.basename(file_path), 0, 0))
                continue
            with f:
                for raw in f:
                    if used >= quota or count >= max_texts:
                        break
                    used += len(raw)   # 按文件真实字节计费
                    try:
                        data = json.loads(raw.decode('utf-8', 'replace'))
                    except Exception:
                        continue
                    text = data.get('text', data.get('content', ''))
                    if text and isinstance(text, str):
                        count += 1
                        yield text
            file_stats.append((os.path.basename(file_path), count, used))
            tqdm.write(f"  读取 {os.path.basename(file_path)}: "
                       f"{count:,} 条 / {used/1e6:.1f} MB")

    print(f"\n开始训练BPE分词器 (词表大小: {vocab_size})...")
    start_time = time.time()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=min_frequency,
        show_progress=True
    )

    tokenizer.train_from_iterator(text_generator(), trainer=trainer)
    strip_bos_eos_added(tokenizer).save(tokenizer_path)

    elapsed = time.time() - start_time
    print(f"\n✓ 分词器训练完成!")
    print(f"  - 词表大小: {tokenizer.get_vocab_size()}")
    print(f"  - 保存路径: {tokenizer_path}")
    print(f"  - 耗时: {timedelta(seconds=int(elapsed))}")
    print(f"  - 使用数据: 按文件大小分配 {max_bytes/1e6:,.0f} MB 预算")
    for name, cnt, used in file_stats:
        print(f"      {name}: {cnt:,} 条 / {used/1e6:,.1f} MB")
    print(f"      (合计 {sum(c for _, c, _ in file_stats):,} 条 / "
          f"{sum(u for _, _, u in file_stats)/1e6:,.1f} MB)")

    print("\n验证特殊token:")
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            print(f"  ✓ {token}: {token_id}")
        else:
            print(f"  ✗ {token}: 缺失!")

    # 质量自检: 中文多字 token 太少说明 merge 预算被英文/代码吃光了，
    # 中文会退化成"一字一 token"（实测 <1% 就是坏的味道）。
    def _is_cjk(c):
        o = ord(c)
        return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF

    single_cjk = multi_cjk = 0
    for tok_str in tokenizer.get_vocab():
        s = tok_str.replace('▁', '')
        if s and all(_is_cjk(c) for c in s):
            if len(s) == 1:
                single_cjk += 1
            else:
                multi_cjk += 1
    vocab = tokenizer.get_vocab_size()
    pct = multi_cjk / vocab if vocab else 0.0
    print(f"\n中文覆盖自检:")
    print(f"  - 单字中文 token: {single_cjk:,}")
    print(f"  - 多字中文 token: {multi_cjk:,} ({pct:.1%})")
    if pct < 0.10:
        print(f"  ⚠️ 多字中文占比过低，中文会接近一字一 token；"
              f"检查语料里中文的占比（当前按文件平分行数）")
    else:
        print(f"  ✓ 中文词表正常")

    return True

# ==================== 分词器 ====================
class BPETokenizer:
    def __init__(self, vocab_size: int = 66536):
        self.vocab_size = vocab_size
        self.special_tokens = SPECIAL_TOKENS
        self.tokenizer = None
        self.special_ids = {}
        # 专门用于“编码正文”的副本：里面去掉 <bos>/<eos> 这两个 added token，
        # 这样文本里字面出现的 "<bos>"/"<eos>" 会走普通 BPE，而不是被识别成
        # 特殊 token id。decode/指纹仍使用原始 self.tokenizer。
        self._plain_tokenizer = None
        self._cache = lru_cache(maxsize=100000)(self._encode)

    def _build_plain_tokenizer(self):
        """构造一个不含 <bos>/<eos> added token 的编码器副本。

        tokenizers 的 added_tokens 会先于 BPE 匹配原文，所以只改 special 标志
        没用；必须把这两项从 added_tokens 里去掉。它们仍保留在 model.vocab
        的 id 2/3 上，因此 bos_token_id/eos_token_id 不受影响。
        decode 继续用原分词器，特殊 token 的显示/跳过行为不变。
        """
        self._plain_tokenizer = None
        if self.tokenizer is None:
            return
        try:
            self._plain_tokenizer = strip_bos_eos_added(self.tokenizer)
        except Exception as e:
            print(f"⚠️ 构建正文编码器失败({e})，回退原分词器："
                  f"文本内的 <bos>/<eos> 仍会被当作特殊 token")
            self._plain_tokenizer = self.tokenizer

    @staticmethod
    def check_existing_tokenizer(tokenizer_path: str = 'tokenizer.json') -> Tuple[bool, dict]:
        if not os.path.exists(tokenizer_path):
            return False, {}
        try:
            temp_tokenizer = Tokenizer.from_file(tokenizer_path)
            vocab_size = temp_tokenizer.get_vocab_size()
            special_tokens_exist = True
            special_ids = {}
            for token in SPECIAL_TOKENS:
                token_id = temp_tokenizer.token_to_id(token)
                if token_id is None:
                    special_tokens_exist = False
                    break
                special_ids[token] = token_id
            info = {
                'vocab_size': vocab_size,
                'special_tokens_exist': special_tokens_exist,
                'special_ids': special_ids,
                'path': tokenizer_path
            }
            return True, info
        except Exception as e:
            print(f"  检查tokenizer时出错: {e}")
            return False, {}

    def _encode(self, text: str) -> List[int]:
        tok = self._plain_tokenizer or self.tokenizer
        return tok.encode(text).ids

    def train(self, texts: List[str], force_retrain: bool = False):
        exists, info = self.check_existing_tokenizer()
        if exists and not force_retrain:
            print(f"✓ 发现已训练的分词器: {info['path']}")
            if info['special_tokens_exist']:
                self.load(info['path'])
                return
            else:
                print(f"  ⚠️ 特殊token不完整，将重新训练...")
        print(f"开始训练新的BPE分词器...")
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        self.tokenizer.decoder = decoders.Metaspace()
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens, min_frequency=2)
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        for token in self.special_tokens:
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                self.special_ids[token] = token_id
        self._build_plain_tokenizer()

    def encode(self, text: str) -> List[int]:
        return self._cache(text)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        tok = self._plain_tokenizer or self.tokenizer
        encoded = tok.encode_batch(texts)
        return [e.ids for e in encoded]

    def decode(self, ids: List[int]) -> str:
        # 与旧版“特殊 token 在 decode 时被跳过”的行为保持一致。移除
        # added token 后，原始 tokenizer 会把 id 2/3 解成字面 "<bos>"/"<eos>"，
        # 这里统一过滤，避免生成结果末尾多出 "<eos>"。
        skip = set(self.special_ids.values())
        if skip:
            ids = [int(i) for i in ids if int(i) not in skip]
        return self.tokenizer.decode(ids)

    def __len__(self):
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0

    @property
    def pad_token_id(self):
        return self.special_ids.get('<pad>', 0)

    @property
    def bos_token_id(self):
        return self.special_ids.get('<bos>', 2)

    @property
    def eos_token_id(self):
        return self.special_ids.get('<eos>', 3)

    def save(self, path: str):
        if self.tokenizer:
            # 与当前 tokenizer.json 一致：保存时不带 <bos>/<eos> added token，
            # 让字面文本按普通 BPE 编码；id 2/3 仍在 model.vocab 里。
            strip_bos_eos_added(self.tokenizer).save(path)

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        for token in self.special_tokens:
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                self.special_ids[token] = token_id
        self._build_plain_tokenizer()
        print(f"✓ 分词器加载成功: {len(self)} 词表"
              f"（文本内字面 <bos>/<eos> 按普通 BPE 编码）")

# ==================== 快速分桶 ====================
class FastBucket:
    __slots__ = ('boundaries', 'max_boundary')
    def __init__(self, boundaries: List[int]):
        self.boundaries = sorted(boundaries)
        self.max_boundary = self.boundaries[-1]

    def get_bucket(self, length: int) -> int:
        if length > self.max_boundary:
            return self.max_boundary
        left, right = 0, len(self.boundaries) - 1
        while left < right:
            mid = (left + right) // 2
            if self.boundaries[mid] < length:
                left = mid + 1
            else:
                right = mid
        return self.boundaries[left]

# ==================== 平铺 uint16 缓存 + memmap 数据集 ====================
CACHE_FORMAT_VERSION = 1

class FlatBucketCache:
    """
    把 <cache_dir>/<cache_id>/bucket_*/chunk_*.pkl 转成
        <hash>/flat_u16/bucket_<b>.tokens.u16   (uint16 平铺 token)
        <hash>/flat_u16/bucket_<b>.offsets.npy (int64, n+1)
        <hash>/flat_u16/manifest.json
    转换是一次性的（本机缓存 7M 样本 / ~5B token，只跑一次）。
    若没有任何 pkl 缓存，则直接从 ./data/*.jsonl 流式编码生成（无需中途落盘 pkl）。
    """
    def __init__(self, cache_id_dir: str, bucket_boundaries: List[int], force: bool = False,
                 data_dir: str = './data', tokenizer=None, max_seq_len: int = 4096,
                 seed: int = 42, max_samples: Optional[int] = None):
        self.src_dir = cache_id_dir
        self.flat_dir = os.path.join(cache_id_dir, 'flat_u16')
        self.manifest_path = os.path.join(self.flat_dir, 'manifest.json')
        self.bucket_boundaries = bucket_boundaries
        self.fast_bucket = FastBucket(bucket_boundaries)
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.max_samples = max_samples
        self.info = {}
        self.source = 'pkl' if self._has_pkl_cache() else 'jsonl'
        if not force and self._load_manifest():
            print(f"✓ 使用平铺缓存: {self.flat_dir}")
        elif self.source == 'pkl':
            self._convert()
        else:
            self._convert_from_jsonl()

    # ---------- 校验 ----------
    def _has_pkl_cache(self) -> bool:
        return bool(glob.glob(os.path.join(self.src_dir, 'bucket_*', 'chunk_*.pkl')))

    def _fingerprint(self):
        if self.source == 'jsonl':
            files = self._jsonl_files()
            return [[os.path.basename(f), os.path.getsize(f)] for f in files]
        fp = {}
        for b in self.bucket_boundaries:
            files = sorted(glob.glob(os.path.join(self.src_dir, f'bucket_{b}', 'chunk_*.pkl')))
            fp[str(b)] = [[os.path.basename(f), os.path.getsize(f)] for f in files]
        return fp

    def _jsonl_files(self):
        files = sorted(glob.glob(os.path.join(self.data_dir, '*.jsonl')))
        return [f for f in files if 'wt.jsonl' not in f and 'bucketed_cache' not in f]

    def _load_manifest(self) -> bool:
        if not os.path.exists(self.manifest_path):
            return False
        try:
            with open(self.manifest_path, 'r') as f:
                m = json.load(f)
            if m.get('version') != CACHE_FORMAT_VERSION:
                return False
            if m.get('bucket_boundaries') != self.bucket_boundaries:
                return False
            # 用 JSONL 指纹校验输入数据是否变化。
            # 兼容两种旧 manifest：
            #   - 有 jsonl_fingerprint 字段：直接比较
            #   - source == 'jsonl'：旧字段 fingerprint 就是 jsonl 指纹
            #   - source == 'pkl'：pkl 缓存本身无法再校验，但 flat 文件完整就继续复用
            jfp = self._jsonl_fingerprint()
            if 'jsonl_fingerprint' in m:
                if m['jsonl_fingerprint'] != jfp:
                    return False
            elif m.get('source', 'pkl') == 'jsonl':
                if m.get('fingerprint') != jfp:
                    return False
            # 缓存是用别的分词器生成的 -> token id 全部失效（旧 manifest 无此字段则放行）
            tfp = self._tokenizer_fingerprint()
            if tfp and 'tokenizer_fingerprint' in m and m['tokenizer_fingerprint'] != tfp:
                print("  ⚠️ 该缓存由不同的分词器生成，token id 不可复用，将重新生成")
                return False
            for b in self.bucket_boundaries:
                info = m['buckets'][str(b)]
                if info['num_samples'] == 0:
                    continue
                tok_path = os.path.join(self.flat_dir, f'bucket_{b}.tokens.u16')
                off_path = os.path.join(self.flat_dir, f'bucket_{b}.offsets.npy')
                if not os.path.exists(tok_path) or not os.path.exists(off_path):
                    return False
                # token 文件大小必须严格等于 num_tokens * 2 (uint16)
                if os.path.getsize(tok_path) != int(info['num_tokens']) * 2:
                    return False
                offs = np.load(off_path, mmap_mode='r')
                if offs.size != int(info['num_samples']) + 1:
                    return False
                if int(offs[-1]) != int(info['num_tokens']):
                    return False
            self.info = m['buckets']
            return True
        except Exception as e:
            print(f"  ⚠️ 平铺缓存校验失败({e})，将重新生成")
            return False

    def _jsonl_fingerprint(self):
        """输入 JSONL 的身份标识（文件名 + 字节大小），用于判断缓存是否可复用。"""
        return [[os.path.basename(f), os.path.getsize(f)] for f in self._jsonl_files()]

    def _tokenizer_fingerprint(self):
        """分词器内容的指纹。token id 的含义完全由它决定，换了分词器缓存必须重建。"""
        return tokenizer_fingerprint(self.tokenizer)

    def _acquire_build_lock(self):
        """同一个缓存目录同时只允许一个进程写入，避免两个 train.py 并发写坏文件。"""
        try:
            import fcntl
        except ImportError:
            return
        os.makedirs(self.src_dir, exist_ok=True)
        self._lock_fh = open(os.path.join(self.src_dir, '.build.lock'), 'w')
        try:
            fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "另一个 train.py 进程正在构建同一个数据缓存，请先停止它再运行。")

    # ---------- 转换 ----------
    def _convert(self):
        import shutil
        self._acquire_build_lock()
        print("\n" + "="*70)
        print("一次性生成快速数据缓存 (uint16 平铺 + offsets, memmap 随机访问)")
        print("="*70)
        if os.path.exists(self.flat_dir):
            shutil.rmtree(self.flat_dir)
        os.makedirs(self.flat_dir, exist_ok=True)

        fingerprint = self._fingerprint()
        buckets_info = {}
        t_start = time.time()
        total_written = 0
        for boundary in tqdm(self.bucket_boundaries, desc="转换分桶", unit="bucket"):
            chunk_files = sorted(glob.glob(os.path.join(self.src_dir, f'bucket_{boundary}', 'chunk_*.pkl')))
            tok_path = os.path.join(self.flat_dir, f'bucket_{boundary}.tokens.u16')
            off_path = os.path.join(self.flat_dir, f'bucket_{boundary}.offsets.npy')

            n_samples = 0
            n_tokens = 0
            all_lengths = []
            with open(tok_path, 'wb') as fo:
                for cf in chunk_files:
                    with open(cf, 'rb') as f:
                        meta = pickle.load(f)
                    samples = meta['samples']           # [n, max_len] int32
                    lengths = meta['lengths'].astype(np.int64)
                    ns = int(samples.shape[0])
                    if ns == 0:
                        continue
                    chunk_tokens = int(lengths.sum())

                    # 去掉 padding，平铺写入（uint16 足够：vocab <= 65536，id 0..65535）
                    flat = np.empty(chunk_tokens, dtype=np.uint16)
                    pos = 0
                    for i in range(ns):
                        l = int(lengths[i])
                        flat[pos:pos + l] = samples[i, :l]
                        pos += l
                    flat.tofile(fo)

                    all_lengths.append(lengths)
                    n_samples += ns
                    n_tokens += chunk_tokens
                    del meta, samples, lengths, flat

            lengths_all = (np.concatenate(all_lengths) if all_lengths
                           else np.zeros(0, dtype=np.int64))
            del all_lengths
            offsets = np.zeros(n_samples + 1, dtype=np.int64)
            np.cumsum(lengths_all, out=offsets[1:])
            np.save(off_path, offsets)
            del lengths_all

            if int(offsets[-1]) != n_tokens:
                raise RuntimeError(f"bucket {boundary} token 数不一致: {int(offsets[-1])} != {n_tokens}"
                                   f"（缓存可能损坏，请删除 {self.flat_dir} 重试）")
            total_written += n_tokens
            buckets_info[str(boundary)] = {
                'num_samples': n_samples,
                'num_tokens': n_tokens,
            }
            print(f"    bucket {boundary:5d}: {n_samples:8,d} 样本 / {n_tokens/1e6:8.1f}M token",
                  flush=True)

        manifest = {
            'version': CACHE_FORMAT_VERSION,
            'source': 'pkl',
            'dtype': 'uint16',
            'bucket_boundaries': self.bucket_boundaries,
            'fingerprint': fingerprint,
            'jsonl_fingerprint': self._jsonl_fingerprint(),
            'tokenizer_fingerprint': self._tokenizer_fingerprint(),
            'buckets': buckets_info,
            'created_at': time.time(),
        }
        tmp = self.manifest_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(manifest, f)
        os.replace(tmp, self.manifest_path)
        self.info = buckets_info
        el = time.time() - t_start
        print(f"✓ 快速缓存生成完成: {total_written/1e9:.2f}B token, "
              f"{sum(v['num_samples'] for v in buckets_info.values()):,} 样本, 耗时 {timedelta(seconds=int(el))}")

    # ---------- 直接从 JSONL 流式编码（没有旧 pkl 缓存时的回退路径） ----------
    def _convert_from_jsonl(self):
        """
        直接从 JSONL 编码生成 flat_u16 缓存。

        与原实现相比：
        1. 使用 tokenizers 的 encode_batch（Rust/Rayon 多线程），比逐条 encode 快 4 倍以上；
        2. 每处理一批就记录 progress.json（输入文件偏移 + 每个桶已写入的字节数），
           中断后从断点继续，不再删除已有结果从头再来；
        3. 最后一个样本处理完才生成 offsets/manifest，生成过程本身也是原子的。
        """
        import shutil

        self._acquire_build_lock()
        print("\n" + "="*70)
        print("一次性生成快速数据缓存 (直接从 JSONL 编码为 uint16 平铺 + offsets)")
        print("="*70)
        if self.tokenizer is None:
            raise ValueError("没有可用缓存且未提供 tokenizer，无法从 JSONL 生成数据")
        jsonl_files = self._jsonl_files()
        if not jsonl_files:
            raise FileNotFoundError(f"{self.data_dir} 下没有找到 *.jsonl 训练数据")

        bnds = list(self.bucket_boundaries)
        fingerprint = [[os.path.basename(f), os.path.getsize(f)] for f in jsonl_files]
        tok_paths = {b: os.path.join(self.flat_dir, f'bucket_{b}.tokens.u16') for b in bnds}
        len_dir = os.path.join(self.flat_dir, 'lengths_u32')
        len_paths = {b: os.path.join(len_dir, f'bucket_{b}.lengths.u32') for b in bnds}
        progress_path = os.path.join(self.flat_dir, 'progress.json')

        state = self._load_progress(progress_path, fingerprint)
        if state is not None and state.get('done') and \
                self._files_match_progress(state, tok_paths, len_paths):
            print(f"✓ 复用已编码的中间结果: {state['total']:,} 样本")
            self._finalize_from_lengths(state, fingerprint, tok_paths, len_paths, progress_path)
            return

        if state is None:
            if os.path.exists(self.flat_dir):
                print(f"  清理旧缓存: {self.flat_dir} (共 {self._dir_size_gb():.2f} GB)")
                shutil.rmtree(self.flat_dir)
            os.makedirs(self.flat_dir, exist_ok=True)
            os.makedirs(len_dir, exist_ok=True)
            state = {
                'version': 1,
                'fingerprint': fingerprint,
                'file_idx': 0,
                'file_pos': 0,
                'total': 0,
                'skipped_long': 0,
                'empty': 0,
                'bad_json': 0,
                'buckets': {
                    str(b): {'num_samples': 0, 'num_tokens': 0,
                             'token_bytes': 0, 'lengths_bytes': 0}
                    for b in bnds
                },
                'done': False,
            }
            for b in bnds:
                open(tok_paths[b], 'wb').close()
                open(len_paths[b], 'wb').close()
            self._save_progress(progress_path, state)
            print("  从第 0 行开始编码")
        else:
            # 断点续传：把未提交的尾部数据截掉，保证文件内容与 progress 一致
            print(f"  发现断点: 文件#{state['file_idx']} 偏移 {int(state['file_pos']):,}，"
                  f"已完成 {state['total']:,} 样本，继续编码...")
            for b in bnds:
                bi = state['buckets'][str(b)]
                self._truncate_file(tok_paths[b], bi['token_bytes'])
                self._truncate_file(len_paths[b], bi['lengths_bytes'])

        tok_fh = {b: open(tok_paths[b], 'ab', buffering=1024 * 1024) for b in bnds}
        len_fh = {b: open(len_paths[b], 'ab', buffering=1024 * 1024) for b in bnds}

        ckpt_sec = float(os.environ.get('CACHE_CKPT_SEC', '20'))
        ckpt_lines = int(os.environ.get('CACHE_CKPT_LINES', '100000'))
        batch_lines = int(os.environ.get('CACHE_BATCH_LINES', '4096'))
        last_ckpt = time.time()
        lines_since_ckpt = 0
        t0 = time.time()
        pbar = tqdm(desc="编码 JSONL", unit="样本", initial=state['total'])

        def commit():
            nonlocal last_ckpt, lines_since_ckpt
            for b in bnds:
                tok_fh[b].flush()
                state['buckets'][str(b)]['token_bytes'] = tok_fh[b].tell()
                len_fh[b].flush()
                state['buckets'][str(b)]['lengths_bytes'] = len_fh[b].tell()
            self._save_progress(progress_path, state)
            last_ckpt = time.time()
            lines_since_ckpt = 0

        try:
            for fi in range(int(state['file_idx']), len(jsonl_files)):
                fp = jsonl_files[fi]
                with open(fp, 'rb') as f:
                    if fi == int(state['file_idx']) and int(state['file_pos']) > 0:
                        f.seek(int(state['file_pos']))
                    while True:
                        if self.max_samples and state['total'] >= self.max_samples:
                            break
                        raw_lines = []
                        for _ in range(batch_lines):
                            raw = f.readline()
                            if not raw:
                                break
                            raw_lines.append(raw)
                        if not raw_lines:
                            break
                        pos_after = f.tell()
                        n_lines = self._encode_batch_into(raw_lines, state, tok_fh, len_fh)
                        del raw_lines
                        # 只有整批写入成功后才推进断点位置
                        state['file_idx'] = fi
                        state['file_pos'] = pos_after
                        pbar.update(n_lines)
                        pbar.set_postfix({'有效': f"{state['total']:,}",
                                          '跳过超长': f"{state['skipped_long']:,}"},
                                         refresh=False)
                        lines_since_ckpt += n_lines
                        if lines_since_ckpt >= ckpt_lines or (time.time() - last_ckpt) >= ckpt_sec:
                            commit()
                if self.max_samples and state['total'] >= self.max_samples:
                    break
                # 当前文件已完整处理
                state['file_idx'] = fi + 1
                state['file_pos'] = 0
                commit()
            # 最后一次提交必须在关闭文件之前完成
            commit()
        finally:
            for h in tok_fh.values():
                h.close()
            for h in len_fh.values():
                h.close()
            pbar.close()

        self._finalize_from_lengths(state, fingerprint, tok_paths, len_paths, progress_path)

    @staticmethod
    def _load_progress(path: str, fingerprint) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            if state.get('version') != 1:
                return None
            if state.get('fingerprint') != fingerprint:
                return None
            if 'buckets' not in state:
                return None
            return state
        except Exception:
            return None

    @staticmethod
    def _save_progress(path: str, state: dict):
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, path)

    @staticmethod
    def _truncate_file(path: str, size):
        size = int(size)
        if not os.path.exists(path):
            open(path, 'wb').close()
            if size > 0:
                raise RuntimeError(f"缓存文件 {path} 丢失，请删除缓存后重新生成")
            return
        cur = os.path.getsize(path)
        if cur > size:
            with open(path, 'r+b') as f:
                f.truncate(size)
        elif cur < size:
            raise RuntimeError(
                f"缓存文件 {path} 比记录短 ({cur} < {size})，请删除缓存后重新生成")

    def _files_match_progress(self, state, tok_paths, len_paths) -> bool:
        try:
            for b in self.bucket_boundaries:
                info = state['buckets'][str(b)]
                if os.path.getsize(tok_paths[b]) != int(info['token_bytes']):
                    return False
                if os.path.getsize(len_paths[b]) != int(info['lengths_bytes']):
                    return False
            return True
        except Exception:
            return False

    def _dir_size_gb(self) -> float:
        total = 0
        for root, _, files in os.walk(self.flat_dir):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total / 1024 ** 3

    def _encode_batch_into(self, raw_lines, state, tok_fh, len_fh) -> int:
        """编码一批原始 JSONL 行，写入各桶并更新 state。返回消费的行数。"""
        texts = []
        n_empty = 0
        n_bad = 0
        for raw in raw_lines:
            line = raw.decode('utf-8', 'replace').strip()
            if not line:
                n_empty += 1
                continue
            try:
                data = json.loads(line)
            except Exception:
                n_bad += 1
                continue
            text = data.get('text', data.get('content', ''))
            if not text or not isinstance(text, str):
                n_empty += 1
                continue
            texts.append(text)

        if texts and self.max_samples:
            remain = self.max_samples - state['total']
            if remain <= 0:
                texts = []
            elif len(texts) > remain:
                texts = texts[:remain]

        # 先在本地累计，全部写盘成功后再更新 state，
        # 这样即使写到一半崩溃/磁盘满，也不会出现“计数已加但 token 没落盘”的错位。
        token_bufs: Dict[int, bytearray] = {}
        length_bufs: Dict[int, array.array] = {}
        delta_samples: Dict[int, int] = {}
        delta_tokens: Dict[int, int] = {}
        n_long = 0
        if texts:
            encodings = self.tokenizer.encode_batch(texts)  # Rust 多线程批量编码
            for ids in encodings:
                l = len(ids)
                if l == 0:
                    n_empty += 1
                    continue
                if l > self.max_seq_len:
                    n_long += 1
                    continue
                b = self.fast_bucket.get_bucket(l)
                token_bufs.setdefault(b, bytearray()).extend(
                    np.asarray(ids, dtype=np.uint16).tobytes())
                length_bufs.setdefault(b, array.array('I')).append(l)
                delta_samples[b] = delta_samples.get(b, 0) + 1
                delta_tokens[b] = delta_tokens.get(b, 0) + l

        for b, buf in token_bufs.items():
            tok_fh[b].write(buf)
        for b, arr in length_bufs.items():
            len_fh[b].write(arr.tobytes())

        for b, ns in delta_samples.items():
            bi = state['buckets'][str(b)]
            bi['num_samples'] += ns
            bi['num_tokens'] += delta_tokens[b]
            state['total'] += ns
        state['skipped_long'] += n_long
        state['empty'] += n_empty
        state['bad_json'] += n_bad
        return len(raw_lines)

    def _finalize_from_lengths(self, state, fingerprint, tok_paths, len_paths, progress_path):
        """根据 lengths_u32 生成 offsets.npy 和 manifest.json。"""
        print("\n整理 offsets ...")
        buckets_info = {}
        total_written = 0
        for b in self.bucket_boundaries:
            info = state['buckets'][str(b)]
            ns = int(info['num_samples'])
            if ns == 0:
                buckets_info[str(b)] = {'num_samples': 0, 'num_tokens': 0}
                continue
            lengths = np.fromfile(len_paths[b], dtype=np.uint32, count=ns).astype(np.int64)
            if lengths.size != ns:
                raise RuntimeError(
                    f"bucket {b} lengths 数量不一致: {lengths.size} != {ns}，请删除缓存重试")
            n_tokens = int(lengths.sum())
            if n_tokens != int(info['num_tokens']):
                raise RuntimeError(
                    f"bucket {b} token 数不一致: {n_tokens} != {info['num_tokens']}，请删除缓存重试")
            actual = os.path.getsize(tok_paths[b])
            if actual != n_tokens * 2:
                raise RuntimeError(
                    f"bucket {b} token 文件大小不一致: {actual} != {n_tokens * 2}，请删除缓存重试")
            offsets = np.zeros(ns + 1, dtype=np.int64)
            np.cumsum(lengths, out=offsets[1:])
            np.save(os.path.join(self.flat_dir, f'bucket_{b}.offsets.npy'), offsets)
            buckets_info[str(b)] = {'num_samples': ns, 'num_tokens': n_tokens}
            total_written += n_tokens

        manifest = {
            'version': CACHE_FORMAT_VERSION,
            'source': 'jsonl',
            'dtype': 'uint16',
            'bucket_boundaries': self.bucket_boundaries,
            'fingerprint': fingerprint,
            'jsonl_fingerprint': fingerprint,
            'tokenizer_fingerprint': self._tokenizer_fingerprint(),
            'buckets': buckets_info,
            'created_at': time.time(),
        }
        tmp = self.manifest_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(manifest, f)
        os.replace(tmp, self.manifest_path)

        state['done'] = True
        self._save_progress(progress_path, state)
        self.info = buckets_info
        print(f"✓ 快速缓存生成完成: {total_written/1e9:.2f}B token, "
              f"{sum(v['num_samples'] for v in buckets_info.values()):,} 样本, "
              f"跳过超长 {state['skipped_long']:,}, 空文本 {state['empty']:,}, "
              f"坏行 {state['bad_json']:,}")

    # ---------- 访问 ----------
    def open_memmaps(self):
        files = {}
        offsets = {}
        for b in self.bucket_boundaries:
            if self.info[str(b)]['num_samples'] == 0:
                files[b] = None
                offsets[b] = None
                continue
            files[b] = np.memmap(os.path.join(self.flat_dir, f'bucket_{b}.tokens.u16'),
                                 dtype=np.uint16, mode='r')
            offsets[b] = np.load(os.path.join(self.flat_dir, f'bucket_{b}.offsets.npy'))
        return files, offsets


class FlatTokenDataset(Dataset):
    """memmap 数据集：__getitem__ 只是一次切片，无 I/O 放大"""
    def __init__(self, cache: FlatBucketCache):
        self.cache = cache
        self.boundaries = list(cache.bucket_boundaries)
        self.files, self.offsets = cache.open_memmaps()

        counts = np.array([cache.info[str(b)]['num_samples'] for b in self.boundaries], dtype=np.int64)
        self.counts = counts
        self.starts = np.zeros(len(self.boundaries), dtype=np.int64)
        np.cumsum(counts[:-1], out=self.starts[1:])
        self.total = int(counts.sum())
        # 每个全局样本所属桶（uint8 足够 84 个桶）
        self.sample_bucket = np.repeat(np.arange(len(self.boundaries), dtype=np.int16), counts)
        self.index2pos = {b: (i, int(self.starts[i])) for i, b in enumerate(self.boundaries)}

    def __len__(self):
        return self.total

    def __getitem__(self, gid: int):
        bi = int(self.sample_bucket[gid])
        b = self.boundaries[bi]
        local = gid - int(self.starts[bi])
        off = self.offsets[b]
        s = int(off[local])
        e = int(off[local + 1])
        return self.files[b][s:e]          # numpy view (uint16)，collate 里做 pad


def collate_pad(batch, eos_id: Optional[int] = None,
                bos_id: Optional[int] = None,
                max_len: Optional[int] = None) -> torch.Tensor:
    """单桶批次 -> [B, Lmax] int64，pad=0。

    缓存的每个样本就是一篇完整文档：
      - bos_id 不为 None：在样本开头加 BOS；
      - eos_id 不为 None：在样本末尾加 EOS（这正是“文档结束”监督）。
    长度超过 max_len 时截断正文，保证 [BOS] + 正文 + [EOS] 不超过 max_len
    （max_seq_len 长度上限）。首 token（BOS）仍只作输入、不产生 loss。
    """
    n = len(batch)
    if n == 0:
        return torch.zeros((0, 0), dtype=torch.long)

    lengths = np.fromiter((a.shape[0] for a in batch),
                          dtype=np.int64, count=n)
    n_bos = 1 if bos_id is not None else 0
    n_eos = 1 if eos_id is not None else 0
    n_special = n_bos + n_eos

    if n_special == 0:
        maxlen = int(lengths.max())
        buf = np.zeros((n, maxlen), dtype=np.int64)
        for i, a in enumerate(batch):
            buf[i, :a.shape[0]] = a
        return torch.from_numpy(buf)

    if max_len is None:
        max_len = int(lengths.max()) + n_special
    new_lengths = np.minimum(lengths + n_special, int(max_len))
    bodies = np.maximum(new_lengths - n_special, 0)   # 保留的真实 token 数
    maxlen = int(new_lengths.max())
    buf = np.zeros((n, maxlen), dtype=np.int64)
    for i, a in enumerate(batch):
        body = int(bodies[i])
        off = 0
        if bos_id is not None:
            buf[i, 0] = bos_id
            off = 1
        if body > 0:
            buf[i, off:off + body] = a[:body]
        if eos_id is not None:
            buf[i, off + body] = eos_id
    return torch.from_numpy(buf)


class TokenBudgetBatchSampler:
    """
    每个优化器 step 严格消费 `samples_per_step`(=batch_size×grad_accum, 当前 64) 个样本；
    每个微批来自单一长度桶，微批大小 = min(max_micro_batch, token_budget // 桶长度)。
    桶的抽样概率与剩余样本数成正比 —— 与旧实现“全局打乱所有 batch”的分布一致。
    """
    def __init__(self, dataset: FlatTokenDataset, samples_per_step: int = 64,
                 target_tokens: int = 12288, max_micro_batch: int = 64, seed: int = 42):
        self.dataset = dataset
        self.samples_per_step = int(samples_per_step)
        self.seed = seed
        self.epoch = 0
        self.skip_samples = 0
        self.boundaries = list(dataset.boundaries)
        self.counts = dataset.counts.copy()
        self.bs_map = {}
        for b in self.boundaries:
            self.bs_map[b] = self._pick_bs(int(b), target_tokens, max_micro_batch, samples_per_step)

    @staticmethod
    def _pick_bs(bucket_len: int, target_tokens: int, max_micro_batch: int, samples_per_step: int) -> int:
        """在 token 预算内选最大的微批，同时避免 step 尾部出现极小的碎批。
        允许 samples_per_step % bs == 0，或尾部至少 max(8, bs//3) 个样本。"""
        bs = max(1, min(max_micro_batch, target_tokens // max(1, bucket_len)))
        while bs > 1:
            rem = samples_per_step % bs
            if rem == 0 or rem >= max(8, bs // 3):
                break
            bs -= 1
        return bs

    def set_epoch(self, epoch: int, skip_samples: int = 0):
        """skip_samples: 本 epoch 已消费的样本数（恢复训练时用来精确跳过）。

        因为采样完全由 (seed + epoch) 决定，重放同样的样本数后 RNG 状态与
        当时逐位一致，后续 batch 既不重复也不遗漏。
        """
        self.epoch = epoch
        self.skip_samples = int(skip_samples)

    def __len__(self):
        return int(math.ceil(self.dataset.total / self.samples_per_step))

    def __iter__(self):
        gen = self._generate()
        skip = int(self.skip_samples)
        if skip > 0:
            skipped = 0
            for batch in gen:
                skipped += len(batch)
                if skipped >= skip:
                    if skipped != skip:
                        raise RuntimeError(
                            f"恢复位置未对齐到 micro-batch 边界: 已跳过 {skipped} 样本，"
                            f"目标 {skip}。checkpoint 可能不是在本实现下保存的。")
                    break
            if skipped != skip:
                raise RuntimeError(f"数据集只有 {skipped} 个样本，无法跳过 {skip} 个")
            print(f"  [resume] 已快进 {skip:,} 个样本到 epoch {self.epoch} 的断点")
        for batch in gen:
            yield batch

    def _generate(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        nb = len(self.boundaries)
        perms = {}
        ptr = {}
        rem = self.counts.copy()
        for i, b in enumerate(self.boundaries):
            n = int(self.counts[i])
            perms[b] = rng.permutation(n) if n > 0 else None
            ptr[b] = 0
        starts = self.dataset.starts
        active = rem > 0
        total_remaining = int(rem.sum())
        budget = self.samples_per_step
        # 选取桶的概率 p_b ∝ rem_b / k_b：
        # 一个桶每次被选中会消费 k_b 个样本，除以 k_b 后各桶的样本消耗速率
        # 与剩余样本数成正比 => 每个样本“下一个被消费”的概率相同，
        # 等价于旧实现的全局 shuffle（不会出现短桶先跑完、长桶垫底）。
        inv_k = np.array([1.0 / self.bs_map[b] for b in self.boundaries], dtype=np.float64)
        weights = np.zeros(nb, dtype=np.float64)
        cum = np.empty(nb, dtype=np.float64)
        while total_remaining > 0:
            have = 0
            while have < budget and total_remaining > 0:
                np.multiply(rem, inv_k, out=weights)
                weights[~active] = 0.0
                np.cumsum(weights, out=cum)
                total_w = float(cum[-1])
                if total_w <= 0.0:
                    break
                r = rng.random() * total_w
                bi = int(np.searchsorted(cum, r, side='right'))
                if bi >= nb:
                    bi = nb - 1
                b = self.boundaries[bi]
                k = min(self.bs_map[b], int(rem[bi]), budget - have)
                if k <= 0:
                    active[bi] = False
                    continue
                p = ptr[b]
                yield (perms[b][p:p + k] + int(starts[bi])).tolist()
                ptr[b] = p + k
                rem[bi] -= k
                total_remaining -= k
                if rem[bi] == 0:
                    active[bi] = False
                have += k


# ==================== 优化版模型组件 ====================
class FusedRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self._use_fused = USE_FUSED_RMSNORM and hasattr(F, 'rms_norm')

    def forward(self, x):
        if self._use_fused:
            # 与 weight * x * rsqrt(mean(x^2)+eps) 完全等价，CUDA 上是单个融合 kernel
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self._precompute(max_len)

    def _precompute(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0))

    def forward(self, seq_len):
        if seq_len > self.cos_cached.shape[1]:
            self._precompute(seq_len)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    head_dim = q.shape[-1]
    if cos.shape[-1] > head_dim:
        cos = cos[..., :head_dim]
        sin = sin[..., :head_dim]
    # 融合 kernel (fast_ops.cu): 一个 kernel 取代 neg*2 + cat*2 + mul*4 + add*2。
    # 前向与 "eager + .half()" 逐位一致；反向只做一次 fp16 舍入 (实测比 eager 更接近 fp64)。
    if _FA_V100_ENABLED and _fa_v100 is not None and head_dim == 64:
        cos2d = cos.reshape(-1, head_dim)
        sin2d = sin.reshape(-1, head_dim)
        if _fa_v100.rope_can_use(q, k, cos2d, sin2d):
            return tuple(_fa_v100.rope(q, k, cos2d, sin2d))
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.wqkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x, mask=None, cos=None, sin=None):
        batch_size, seq_len, _ = x.shape

        qkv = self.wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 训练/推理都用 causal、无 attn_mask 的 SDPA 语义；能喂给 sm70 kernel 就直接用，
        # 否则回退（例如传了自定义 mask、或 dtype/head_dim/对齐不满足）。
        attn = None
        if mask is None:
            fq, fk, fv = _fa_v100_half_inputs(q, k, v)
            if _fa_v100_usable(fq, fk, fv):
                attn = _fa_v100.flash_attn_v100(
                    fq, fk, fv,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=True,
                )
        if attn is None:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.wo(attn)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))

class DecoderOnlyBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = FusedRMSNorm(config.d_model)
        self.norm2 = FusedRMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None, cos=None, sin=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(x, mask=mask, cos=cos, sin=sin)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        if config.use_rope:
            self.rope = RotaryPositionalEmbedding(config.d_model, config.max_seq_len)

        self.layers = nn.ModuleList([DecoderOnlyBlock(config) for _ in range(config.n_layers)])
        self.norm = FusedRMSNorm(config.d_model)
        self.fc_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.fc_out.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self):
        self.use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.use_gradient_checkpointing = False

    def _custom_forward(self, layer, x, mask, cos, sin):
        return layer(x, mask=mask, cos=cos, sin=sin)

    def forward_hidden(self, x, mask=None):
        batch_size, seq_len = x.shape

        x = self.embedding(x)

        cos, sin = None, None
        if self.config.use_rope:
            cos, sin = self.rope(seq_len)
            cos = cos.to(x.dtype)
            sin = sin.to(x.dtype)

        if self.use_gradient_checkpointing and self.training:
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(
                    self._custom_forward, layer, x, mask, cos, sin,
                    use_reentrant=False
                )
        else:
            for layer in self.layers:
                x = layer(x, mask=mask, cos=cos, sin=sin)

        return self.norm(x)

    def forward(self, x, mask=None):
        return self.fc_out(self.forward_hidden(x, mask))

    def compute_loss(self, input_ids, labels=None, ignore_index=0, chunk_rows: int = 2048):
        """与旧实现数学等价：mean over 非 ignore token 的 shifted CE。
        改为分块计算，避免物化 [B*(S-1), vocab] 的 fp32 logits 及 contiguous 副本。"""
        if labels is None:
            labels = input_ids

        hidden = self.forward_hidden(input_ids)
        hidden = hidden[:, :-1, :]
        target = labels[:, 1:]

        n_tokens = (target != ignore_index).sum().clamp(min=1)
        d_model = hidden.shape[-1]
        hidden = hidden.reshape(-1, d_model)
        target = target.reshape(-1)

        total = None
        for i in range(0, hidden.shape[0], chunk_rows):
            logits = self.fc_out(hidden[i:i + chunk_rows])
            part = F.cross_entropy(logits, target[i:i + chunk_rows],
                                   ignore_index=ignore_index, reduction='sum')
            total = part if total is None else total + part
        if total is None:
            return torch.zeros((), device=input_ids.device, requires_grad=True)
        return total / n_tokens

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def save(self, path: str):
        ckpt = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }
        # 让 inf.py 知道这份权重训练时是否用了 BOS/EOS
        if hasattr(self, 'prepend_bos'):
            ckpt['bos_supervised'] = bool(self.prepend_bos)
        if hasattr(self, 'append_eos'):
            ckpt['eos_supervised'] = bool(self.append_eos)
        torch.save(ckpt, path)
        print(f"✓ 模型已保存到: {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        print(f"✓ 模型已从 {path} 加载")

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 base_lr: float, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = base_lr * min_lr_ratio
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = min(1.0, (self.current_step - self.warmup_steps) /
                          max(1, (self.total_steps - self.warmup_steps)))
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

# ==================== 训练函数 ====================
def train_model_optimized(model, train_loader, config: TrainingConfig, save_dir='./checkpoints'):
    global _current_model, _current_optimizer, _current_scaler, _current_scheduler
    global _current_epoch, _current_step, _interrupted
    global _save_dir
    _save_dir = save_dir

    use_fp16 = config.use_fp16 and torch.cuda.is_available() and supports_fp16
    use_compile = (config.use_compile and supports_compile
                   and torch.cuda.is_available()
                   and os.environ.get('UNIVERSAL_LM_COMPILE', os.environ.get('EMINI_COMPILE', '1')) != '0')

    seq_len = model.config.max_seq_len
    samples_per_step = config.batch_size * config.gradient_accumulation_steps
    print(f"\n{'='*70}")
    print(f"Training Configuration (Universal LM Trainer - Max Seq Len = {seq_len})")
    print(f"{'='*70}")
    print(f"Mode: {'🧪 TEST MODE' if config.test_mode else '🚀 FULL TRAINING'}")
    print(f"每步样本数 (effective batch): {samples_per_step}")
    print(f"微批 token 预算: {config.target_tokens}")
    print(f"Max Sequence Length: {seq_len}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Warmup Steps: {config.warmup_steps}")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'} (Tensor Cores)")
    print(f"torch.compile: {'启用' if use_compile else '禁用'}")
    print(f"DataLoader Workers: {config.num_workers}")
    print(f"Gradient Checkpointing: {'启用' if model.config.use_gradient_checkpointing else '禁用'}")
    print(f"进度保存: 手动（Ctrl+C 菜单 1/3/6）-> {os.path.join(save_dir, 'last.pt')}")
    print(f"普通检查点: Ctrl+C 菜单 7 -> {os.path.join(save_dir, 'checkpoint_epoch{E}_step{S}.pt')}")
    print(f"恢复训练: {config.resume_path or '否（从头开始）'}")
    print(f"{'='*70}\n")

    if use_compile and not config.test_mode:
        # 编译 forward_hidden（训练真正走的路径；compute_loss 直接调用它）。
        # 只编译这一层：attention 前后的 norm/add/cast/silu*mul/dropout 会被融合成
        # 少量 kernel，attention 本身（自定义 autograd.Function）自然断图。
        # 不用 max-autotune/fullgraph：max-autotune 在 V100 上只会浪费启动时间，
        # fullgraph 则会因为自定义 attention 算子直接失败。
        try:
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 512
            model.forward_hidden = torch.compile(model.forward_hidden, dynamic=True)
            print("✓ torch.compile 已启用 (forward_hidden, dynamic=True)")
        except Exception as e:
            print(f"⚠️ 编译失败: {e}")
            use_compile = False

    model.train()
    # 供 Ctrl+C 菜单里的生成测试 / model.save() 记录 BOS/EOS 设置
    model.prepend_bos = bool(config.prepend_bos)
    model.append_eos = bool(config.append_eos)
    _orig = get_original_model(model)
    _orig.prepend_bos = bool(config.prepend_bos)
    _orig.append_eos = bool(config.append_eos)

    optim_kwargs = dict(lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.95))
    optimizer = None
    if config.fused_optimizer and torch.cuda.is_available():
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **optim_kwargs)
            print("✓ 使用 fused AdamW")
        except Exception as e:
            print(f"⚠️ fused AdamW 不可用({e})，回退默认实现")
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)

    total_samples = len(train_loader.dataset)
    effective_batch_size = samples_per_step
    steps_per_epoch = (total_samples + effective_batch_size - 1) // effective_batch_size
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = min(config.warmup_steps, max(1, total_steps // 10))

    print(f"\nDataset Statistics:")
    print(f"  - Total samples: {total_samples:,}")
    print(f"  - Steps per epoch: {steps_per_epoch:,}")
    print(f"  - Total steps: {total_steps:,}")
    print(f"  - Warmup steps: {warmup_steps}")

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=config.learning_rate,
        min_lr_ratio=config.min_lr_ratio
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16) if hasattr(torch, 'amp') else torch.cuda.amp.GradScaler(enabled=use_fp16)
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    # ---- 数据指纹：恢复时保证 sample id 语义完全一致 ----
    ds = train_loader.dataset
    cache_obj = getattr(ds, 'cache', None)
    try:
        counts = np.ascontiguousarray(getattr(ds, 'counts', np.array([], dtype=np.int64)),
                                      dtype=np.int64)
    except Exception:
        counts = np.array([], dtype=np.int64)
    data_fingerprint = {
        'cache_id': os.path.basename(os.path.dirname(getattr(cache_obj, 'flat_dir', '')))
                    if cache_obj is not None else None,
        'tokenizer_fp': cache_obj._tokenizer_fingerprint() if cache_obj is not None else None,
        'total_samples': int(getattr(ds, 'total', 0)),
        'counts_md5': hashlib.md5(counts.tobytes()).hexdigest(),
        'boundaries': list(getattr(ds, 'boundaries', [])),
        'samples_per_step': int(samples_per_step),
        # 这两个参数决定 bs_map => 决定每步的桶选择与 RNG 消耗，必须一起校验，
        # 否则改预算后续训会静默错位（重复/遗漏样本）。
        'target_tokens': int(config.target_tokens),
        'max_micro_batch': int(config.max_micro_batch),
        'seed': int(config.seed),
        'sampler_version': SAMPLER_VERSION,
        # BOS/EOS 监督会改变每个样本的 target 数（但不改变样本顺序）。必须写进
        # 指纹：从“无 BOS/EOS”的 checkpoint 续训时缺这些字段会告警并按当前值
        # 继续；一旦保存过新 checkpoint，再切回 --no-eos/--no-bos 就会明确拒绝。
        'append_eos': bool(config.append_eos),
        'eos_id': (int(model.tokenizer.eos_token_id)
                   if config.append_eos else None),
        'prepend_bos': bool(config.prepend_bos),
        'bos_id': (int(model.tokenizer.bos_token_id)
                   if config.prepend_bos else None),
    }

    # 进度文件路径
    global _current_save_path, _TRAIN_STATE
    _current_save_path = os.path.join(save_dir, 'last.pt')
    _TRAIN_STATE = None

    resume_info = None
    if config.resume_path:
        resume_info = load_training_checkpoint(
            config.resume_path, model, optimizer, scaler, scheduler,
            data_fingerprint, device
        )
        global_step = resume_info['global_step']
        best_loss = resume_info['best_loss']
        start_epoch = resume_info['epoch']
        start_step_in_epoch = resume_info['step_in_epoch']
        start_epoch_samples = resume_info['epoch_samples']
        start_epoch_loss_sum = resume_info['epoch_loss_sum']

        # checkpoint 若是“epoch 最后一 step”存的，直接进入下一 epoch
        if start_epoch_samples >= total_samples:
            start_epoch += 1
            start_step_in_epoch = 0
            start_epoch_samples = 0
            start_epoch_loss_sum = 0.0

        if start_epoch > config.epochs:
            print(f"\n✓ checkpoint 显示训练已全部完成 "
                  f"({config.epochs} epochs / {global_step} steps)，无需继续。")
            return best_loss

        if global_step >= total_steps:
            print(f"\n✓ 已达到总步数 {total_steps}，无需继续。")
            return best_loss

        print(f"  - 恢复位置: epoch {start_epoch}/{config.epochs}, "
              f"epoch 内 step {start_step_in_epoch}/{steps_per_epoch}, "
              f"本 epoch 已消费 {start_epoch_samples:,}/{total_samples:,} 样本")
    else:
        global_step = 0
        start_epoch = 1
        start_step_in_epoch = 0
        start_epoch_samples = 0
        start_epoch_loss_sum = 0.0

    _current_model = model
    _current_optimizer = optimizer
    _current_scaler = scaler
    _current_scheduler = scheduler

    training_start_time = time.time()
    pad_id = model.tokenizer.pad_token_id

    step_times = deque(maxlen=100)
    token_rates = deque(maxlen=100)
    def make_autocast():
        if hasattr(torch, 'amp'):
            return torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_fp16)
        return torch.cuda.amp.autocast(enabled=use_fp16, dtype=torch.float16)

    for epoch in range(start_epoch, config.epochs + 1):
        _current_epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sampler = getattr(train_loader, 'batch_sampler', None)

        is_resume_epoch = (epoch == start_epoch and resume_info is not None)
        epoch_skip = start_epoch_samples if is_resume_epoch else 0
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch, skip_samples=epoch_skip)

        epoch_start_time = time.time()
        epoch_loss = start_epoch_loss_sum if is_resume_epoch else 0.0
        step_in_epoch = start_step_in_epoch if is_resume_epoch else 0
        epoch_samples_consumed = start_epoch_samples if is_resume_epoch else 0
        accum_samples = 0
        loss_accum = None
        last_step_start = time.time()
        tokens_in_step = 0

        pbar = tqdm(total=steps_per_epoch, initial=step_in_epoch,
                    desc=f"Epoch {epoch}/{config.epochs}",
                    unit='step', dynamic_ncols=True)
        stop_training = False

        try:
            for input_ids in train_loader:
                if check_interrupt():
                    action = show_interrupt_menu()
                    if action != 'continue':
                        break

                n = input_ids.shape[0]
                tokens_in_step += input_ids.shape[1] * n
                input_ids = input_ids.to(device, non_blocking=True)

                with make_autocast():
                    loss = model.compute_loss(input_ids, ignore_index=pad_id,
                                              chunk_rows=config.ce_chunk_rows)

                # 按样本数加权 —— 等价于旧实现 loss/G（旧实现每个微批样本数相同）
                w = n / samples_per_step
                scaler.scale(loss * w).backward()
                loss_accum = loss.detach() * w if loss_accum is None else loss_accum + loss.detach() * w
                accum_samples += n
                epoch_samples_consumed += n

                if accum_samples >= samples_per_step:
                    if use_fp16:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        optimizer.step()

                    current_lr = scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    step_loss = float(loss_accum.item())
                    loss_accum = None
                    accum_samples = 0
                    epoch_loss += step_loss
                    step_in_epoch += 1
                    global_step += 1
                    _current_step = global_step

                    # 只有在此刻（优化器已更新、samples_per_step 个样本全部消费完）快照，
                    # 才能保证 checkpoint 与样本消费位置严格对齐。
                    # 不自动落盘；由 Ctrl+C 菜单显式保存。
                    _snapshot_train_state(epoch, step_in_epoch, global_step,
                                          epoch_samples_consumed, epoch_loss,
                                          best_loss, data_fingerprint)

                    now = time.time()
                    step_time = now - last_step_start
                    last_step_start = now
                    step_times.append(step_time)
                    token_rates.append(tokens_in_step / max(step_time, 1e-6))
                    tokens_in_step = 0

                    avg_step = sum(step_times) / len(step_times)
                    avg_rate = sum(token_rates) / len(token_rates)
                    if torch.cuda.is_available():
                        mem_used = torch.cuda.memory_allocated() / 1e9
                        mem_reserved = torch.cuda.memory_reserved() / 1e9
                        mem_info = f'{mem_used:.1f}/{mem_reserved:.1f}G'
                    else:
                        mem_info = 'N/A'

                    pbar.set_postfix({
                        'loss': f'{step_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'step': f'{avg_step*1000:.0f}ms',
                        'tok/s': f'{avg_rate/1000:.1f}k',
                        'mem': mem_info,
                    })
                    pbar.update(1)

                    if step_in_epoch % config.log_every == 0:
                        tqdm.write(
                            f"\n  [Epoch {epoch}] Step {step_in_epoch}/{steps_per_epoch}: "
                            f"loss={step_loss:.4f}, step={avg_step*1000:.0f}ms, "
                            f"tok/s={avg_rate/1000:.1f}k, mem={mem_info}, lr={current_lr:.2e}"
                        )

                    if config.max_steps and global_step >= config.max_steps:
                        stop_training = True
                        break

            # epoch 末尾不足 samples_per_step 样本
            if accum_samples > 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_in_epoch += 1
                global_step += 1
                _current_step = global_step
                if loss_accum is not None:
                    epoch_loss += float(loss_accum.item())
                # 本 epoch 的最后一个（不足 samples_per_step 样本的）step 也落一次快照
                _snapshot_train_state(epoch, step_in_epoch, global_step,
                                      epoch_samples_consumed, epoch_loss,
                                      best_loss, data_fingerprint)
                accum_samples = 0
                loss_accum = None

        except KeyboardInterrupt:
            print("\n\n⚠️ 训练被中断")
            action = show_interrupt_menu()
            if action == 'continue':
                continue
            else:
                break
        except Exception as e:
            print(f"\n训练出错: {e}")
            import traceback
            traceback.print_exc()
            break

        pbar.close()

        if step_in_epoch == 0:
            continue

        avg_loss = epoch_loss / step_in_epoch
        epoch_time = time.time() - epoch_start_time

        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        avg_rate = sum(token_rates) / len(token_rates) if token_rates else 0

        print(f"\n📊 Epoch {epoch} 完成:")
        print(f"  - Loss: {avg_loss:.4f}")
        print(f"  - 平均Step时间: {avg_step_time*1000:.1f}ms")
        print(f"  - 吞吐量: {avg_rate/1000:.1f}k token/s ({step_in_epoch / epoch_time:.2f} step/s)")
        if torch.cuda.is_available():
            print(f"  - 显存使用: {torch.cuda.memory_allocated()/1e9:.2f}GB "
                  f"(峰值 {torch.cuda.max_memory_allocated()/1e9:.2f}GB)")
        print(f"  - Epoch耗时: {epoch_time:.1f}秒")

        step_times.clear()
        token_rates.clear()

        model_to_save = get_original_model(model)
        skip_save = stop_training or config.max_steps > 0
        if avg_loss < best_loss and not config.test_mode and not skip_save:
            best_loss = avg_loss
            model_to_save.save(os.path.join(save_dir, 'best_model.pt'))

        if epoch % config.save_every == 0 and not config.test_mode and not skip_save:
            model_to_save.save(os.path.join(save_dir, f'model_epoch_{epoch}.pt'))

        # 本 epoch 已完整跑完：把恢复指针移到下一个 epoch 起点（skip=0）
        # （这是边界保存，不是定期保存；step 级保存只由 Ctrl+C 菜单触发）
        if not stop_training and not config.test_mode and config.max_steps == 0:
            _snapshot_train_state(epoch + 1, 0, global_step, 0, 0.0,
                                  best_loss, data_fingerprint)
            save_checkpoint()

        if stop_training or (config.max_steps and global_step >= config.max_steps):
            print(f"\n已达到 max_steps={config.max_steps}，停止。")
            break

    total_time = time.time() - training_start_time
    print(f"\n✓ 训练完成! 总耗时: {timedelta(seconds=int(total_time))}")
    return best_loss

# ==================== 主程序 ====================
DEFAULT_CONFIG = {
    "model": {
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 10,
        "d_ff": 3072,
        "max_seq_len": 4096,
        "dropout": 0.1,
        "use_rope": True,
        "tie_weights": True,
        "use_gradient_checkpointing": False,
    },
    "training": {
        "batch_size": 64,
        "gradient_accumulation_steps": 1,
        "epochs": 1,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "warmup_steps": 3000,
        "min_lr_ratio": 0.15,
        "save_every": 1,
        "log_every": 50,
        "num_workers": 8,
        "prefetch_factor": 4,
        "seed": 42,
        "use_fp16": True,
        "use_compile": True,
        "fused_optimizer": True,
        "ce_chunk_rows": 2048,
        "target_tokens": 12288,
        "max_micro_batch": 64,
        "test_samples": 100000,
        "prepend_bos": True,
        "append_eos": True,
        "max_steps": 0,
    },
    "tokenizer": {
        "vocab_size": 65536,
        "path": "tokenizer.json",
        "max_bytes": 300000000,
        "min_frequency": 2,
    },
    "paths": {
        "data_dir": "data",
        "cache_dir": "cache",
        "checkpoint_dir": "checkpoints",
    },
}

BUCKET_STEP = 50
UINT16_LIMIT = 65536


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_project_config(path: str) -> dict:
    cfg = deep_merge({}, DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)
        cfg = deep_merge(cfg, user_cfg)
        print(f"✓ 加载配置: {os.path.abspath(path)}")
    else:
        print(f"⚠️ 未找到配置文件 {path}，使用内置默认值")
    return cfg


def build_bucket_boundaries(max_seq_len: int) -> List[int]:
    """按固定步长生成长度分桶边界，最后一桶严格等于 max_seq_len。"""
    if max_seq_len < BUCKET_STEP:
        return [max_seq_len]
    bounds = list(range(BUCKET_STEP, max_seq_len, BUCKET_STEP))
    bounds.append(max_seq_len)
    return bounds


def parse_args():
    p = argparse.ArgumentParser(
        description="通用 Decoder-only 语言模型训练脚本（配置驱动）")
    p.add_argument('--config', type=str, default='config.json',
                   help='配置文件路径（默认 config.json）')
    p.add_argument('--data-dir', type=str, default=None,
                   help='训练数据目录（覆盖配置）')
    p.add_argument('--cache-dir', type=str, default=None,
                   help='缓存根目录（覆盖配置）')
    p.add_argument('--checkpoint-dir', type=str, default=None,
                   help='checkpoint 目录（覆盖配置）')
    p.add_argument('--tokenizer', type=str, default=None,
                   help='tokenizer.json 路径（覆盖配置）')
    p.add_argument('--resume', nargs='?', const='__AUTO__', default=None,
                   help='从 checkpoint 恢复；不带路径时用 <checkpoint_dir>/last.pt')
    p.add_argument('--max-steps', type=int, default=None,
                   help='只训练指定 step 数（调试/基准）')
    p.add_argument('--retrain-tokenizer', action='store_true',
                   help='强制重新训练 tokenizer（会改变 token id）')
    p.add_argument('--cache-only', action='store_true',
                   help='只构建数据缓存后退出')
    p.add_argument('--no-bos', action='store_true', help='关闭 BOS 监督')
    p.add_argument('--no-eos', action='store_true', help='关闭 EOS 监督')
    p.add_argument('--test-mode', action='store_true',
                   help='只使用少量样本跑通流程')
    p.add_argument('--seed', type=int, default=None, help='随机种子（覆盖配置）')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_project_config(args.config)

    model_cfg = cfg['model']
    train_cfg = cfg['training']
    tok_cfg = cfg['tokenizer']
    path_cfg = cfg['paths']

    if args.data_dir:
        path_cfg['data_dir'] = args.data_dir
    if args.cache_dir:
        path_cfg['cache_dir'] = args.cache_dir
    if args.checkpoint_dir:
        path_cfg['checkpoint_dir'] = args.checkpoint_dir
    if args.tokenizer:
        tok_cfg['path'] = args.tokenizer
    if args.seed is not None:
        train_cfg['seed'] = args.seed

    data_dir = path_cfg['data_dir']
    cache_root = path_cfg['cache_dir']
    checkpoint_dir = path_cfg['checkpoint_dir']
    tokenizer_path = tok_cfg['path']

    max_seq_len = int(model_cfg['max_seq_len'])
    vocab_size_cfg = int(tok_cfg['vocab_size'])
    seed = int(train_cfg['seed'])

    if vocab_size_cfg > UINT16_LIMIT:
        raise SystemExit(
            f"❌ tokenizer.vocab_size={vocab_size_cfg} 超过 uint16 缓存上限 "
            f"{UINT16_LIMIT}；请调小词表，或修改缓存格式后重试。")
    if max_seq_len <= 0:
        raise SystemExit("❌ model.max_seq_len 必须大于 0")
    if int(model_cfg['d_model']) % int(model_cfg['n_heads']) != 0:
        raise SystemExit("❌ model.d_model 必须能被 model.n_heads 整除")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(cache_root, exist_ok=True)

    # ---- resume 路径 ----
    resume_path = None
    if args.resume is not None:
        resume_path = (os.path.join(checkpoint_dir, 'last.pt')
                       if args.resume == '__AUTO__' else args.resume)

    print("=" * 70)
    print("Universal LM Trainer - Decoder-only 语言模型预训练")
    print("=" * 70)
    print(f"配置: d_model={model_cfg['d_model']} layers={model_cfg['n_layers']} "
          f"heads={model_cfg['n_heads']} d_ff={model_cfg['d_ff']} "
          f"max_seq_len={max_seq_len} vocab={vocab_size_cfg}")
    print(f"数据: {os.path.abspath(data_dir)}")
    print(f"缓存: {os.path.abspath(cache_root)}")
    print(f"检查点: {os.path.abspath(checkpoint_dir)}")

    if resume_path and not os.path.exists(resume_path):
        print(f"❌ 指定了 --resume，但找不到进度文件: {resume_path}")
        print("   如果要从头训练，请不要加 --resume")
        return
    if resume_path and args.retrain_tokenizer:
        print("❌ --resume 与 --retrain-tokenizer 冲突：重训 tokenizer 会改变所有 token id，")
        print("   无法与旧进度拼接。要么去掉 --resume 从头训练，要么不要重训 tokenizer。")
        return
    if resume_path and not os.path.exists(tokenizer_path):
        print(f"❌ 要恢复训练，但 tokenizer 不存在: {tokenizer_path}")
        print("   没有原 tokenizer 就无法复现 token id，不能安全恢复；请从头训练。")
        return

    # ==================== 分词器 ====================
    print("\n" + "=" * 70)
    print("步骤 1: 训练/加载分词器")
    print("=" * 70)

    if not os.path.isdir(data_dir):
        print(f"❌ 数据目录 {data_dir} 不存在!")
        print("请创建数据目录并放入 JSONL 训练数据（字段 text 或 content）")
        return

    retrain_tokenizer = args.retrain_tokenizer
    need_train = retrain_tokenizer
    if retrain_tokenizer:
        print("--retrain-tokenizer 指定: 强制重新训练分词器")
        if os.path.exists(tokenizer_path):
            backup = tokenizer_path + '.bak'
            shutil.copy2(tokenizer_path, backup)
            print(f"  旧分词器已备份到 {backup}")
    elif not os.path.exists(tokenizer_path):
        print(f"未找到分词器 {tokenizer_path}，将训练新分词器 "
              f"(按文件大小分配 {tok_cfg['max_bytes']/1e6:,.0f} MB 语料)")
        need_train = True
    else:
        try:
            temp_tokenizer = Tokenizer.from_file(tokenizer_path)
            for token in SPECIAL_TOKENS:
                if temp_tokenizer.token_to_id(token) is None:
                    print(f"⚠️ 分词器中缺少特殊 token: {token}")
                    need_train = True
                    break
            if not need_train:
                print(f"✓ 使用已存在的分词器: {tokenizer_path}")
        except Exception as e:
            print(f"⚠️ 读取分词器失败: {e}")
            need_train = True

    if need_train:
        print(f"\n开始训练 BPE 分词器 (vocab={vocab_size_cfg}, "
              f"语料预算 {tok_cfg['max_bytes']/1e6:,.0f} MB)...")
        success = train_tokenizer(
            data_dir=data_dir,
            vocab_size=vocab_size_cfg,
            tokenizer_path=tokenizer_path,
            min_frequency=int(tok_cfg['min_frequency']),
            force_retrain=retrain_tokenizer,
            max_bytes=int(tok_cfg['max_bytes']),
        )
        if not success:
            print("❌ 分词器训练失败!")
            return
        print("✓ 分词器训练完成!")
    else:
        print("✓ 使用已有分词器")

    tokenizer = BPETokenizer(vocab_size=vocab_size_cfg)
    tokenizer.load(tokenizer_path)

    if len(tokenizer) > UINT16_LIMIT:
        raise ValueError(
            f"词表大小 {len(tokenizer)} 超过 uint16 缓存上限 {UINT16_LIMIT}，"
            f"请调小 tokenizer.vocab_size")

    # ==================== 训练配置 ====================
    append_eos = bool(train_cfg['append_eos']) and not args.no_eos
    prepend_bos = bool(train_cfg['prepend_bos']) and not args.no_bos
    max_steps = int(train_cfg.get('max_steps', 0) or 0)
    if args.max_steps is not None:
        max_steps = args.max_steps

    train_config = TrainingConfig(
        batch_size=int(train_cfg['batch_size']),
        epochs=int(train_cfg['epochs']),
        learning_rate=float(train_cfg['learning_rate']),
        weight_decay=float(train_cfg['weight_decay']),
        gradient_accumulation_steps=int(train_cfg['gradient_accumulation_steps']),
        max_grad_norm=float(train_cfg['max_grad_norm']),
        save_every=int(train_cfg['save_every']),
        log_every=int(train_cfg['log_every']),
        num_workers=int(train_cfg['num_workers']),
        seed=seed,
        prefetch_factor=int(train_cfg['prefetch_factor']),
        use_fp16=bool(train_cfg['use_fp16']),
        use_compile=bool(train_cfg['use_compile']),
        warmup_steps=int(train_cfg['warmup_steps']),
        min_lr_ratio=float(train_cfg['min_lr_ratio']),
        test_mode=bool(args.test_mode),
        test_samples=int(train_cfg['test_samples']),
        target_tokens=int(train_cfg['target_tokens']),
        max_micro_batch=int(train_cfg['max_micro_batch']),
        fused_optimizer=bool(train_cfg['fused_optimizer']),
        ce_chunk_rows=int(train_cfg['ce_chunk_rows']),
        max_steps=max_steps,
        resume_path=resume_path,
        append_eos=append_eos,
        prepend_bos=prepend_bos,
    )

    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    bucket_boundaries = build_bucket_boundaries(max_seq_len)

    # cache id 只由 tokenizer / 序列长度 / 分桶 / 采样种子决定，与模型结构无关，
    # 因此换模型大小可以直接复用同一份缓存。
    hash_input = f"{len(tokenizer)}_{max_seq_len}_None_{train_config.seed}_5000"
    for b in bucket_boundaries:
        hash_input += f"_{b}"
    cache_id = hashlib.md5(hash_input.encode()).hexdigest()[:16]
    cache_id_dir = os.path.join(cache_root, cache_id)

    if resume_path and os.path.exists(resume_path + '.json'):
        try:
            with open(resume_path + '.json', 'r') as f:
                _meta = json.load(f)
            _saved_fp = _meta.get('data_fingerprint') or {}
            _cur_tfp = tokenizer_fingerprint(tokenizer)
            _problems = []
            if _saved_fp.get('tokenizer_fp') and _saved_fp['tokenizer_fp'] != _cur_tfp:
                _problems.append(
                    f"tokenizer: checkpoint={_saved_fp['tokenizer_fp']} 当前={_cur_tfp}")
            if _saved_fp.get('cache_id') and _saved_fp['cache_id'] != cache_id:
                _problems.append(
                    f"cache_id: checkpoint={_saved_fp['cache_id']} 当前={cache_id}")
            if _problems:
                print("\n❌ 训练进度与当前 tokenizer/缓存不匹配，拒绝恢复：")
                for p in _problems:
                    print(f"   - {p}")
                print("   换了 tokenizer 或缓存后必须从头训练（不要加 --resume）。")
                return
            print(f"✓ 进度文件预检通过: epoch={_meta.get('epoch')} "
                  f"global_step={_meta.get('global_step')}")
        except Exception as e:
            print(f"⚠️ 读取进度元数据失败（将走完整校验）: {e}")

    if retrain_tokenizer and os.path.exists(cache_id_dir):
        stale = cache_id_dir + datetime.now().strftime('.old-%Y%m%d-%H%M%S')
        os.rename(cache_id_dir, stale)
        print(f"⚠️ 分词器已重训，旧 token 缓存改名为: {os.path.basename(stale)}")

    print(f"\n数据缓存目录: {cache_id_dir}")
    flat_cache = FlatBucketCache(
        cache_id_dir, bucket_boundaries,
        data_dir=data_dir, tokenizer=tokenizer, max_seq_len=max_seq_len,
        seed=train_config.seed,
        max_samples=train_config.test_samples if train_config.test_mode else None,
    )

    dataset = FlatTokenDataset(flat_cache)
    print(f"✓ 数据集就绪: {len(dataset):,} 样本, "
          f"{sum(v['num_tokens'] for v in flat_cache.info.values())/1e9:.2f}B token")
    if args.cache_only:
        print("\n--cache-only: 数据缓存已生成，退出（不开始训练）。")
        return

    batch_sampler = TokenBudgetBatchSampler(
        dataset,
        samples_per_step=(train_config.batch_size *
                          train_config.gradient_accumulation_steps),
        target_tokens=train_config.target_tokens,
        max_micro_batch=train_config.max_micro_batch,
        seed=train_config.seed,
    )
    print("  微批大小表 (桶长度 -> 样本数):")
    for b in bucket_boundaries[::max(1, len(bucket_boundaries) // 14)]:
        print(f"    {b:5d} -> {batch_sampler.bs_map[b]:3d}")

    loader_kwargs = dict(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(
            collate_pad,
            eos_id=(tokenizer.eos_token_id if train_config.append_eos else None),
            bos_id=(tokenizer.bos_token_id if train_config.prepend_bos else None),
            max_len=max_seq_len,
        ),
    )
    if train_config.num_workers > 0:
        loader_kwargs['prefetch_factor'] = train_config.prefetch_factor
        loader_kwargs['persistent_workers'] = True
    train_loader = DataLoader(**loader_kwargs)
    if train_config.append_eos or train_config.prepend_bos:
        parts = []
        if train_config.prepend_bos:
            parts.append(f"开头加 <bos> id={tokenizer.bos_token_id}")
        if train_config.append_eos:
            parts.append(f"末尾加 <eos> id={tokenizer.eos_token_id}")
        print("✓ BOS/EOS 监督: 开（" + "；".join(parts) +
              f"；超长样本截断正文以保持宽度 ≤{max_seq_len}）")
    else:
        print("✓ BOS/EOS 监督: 关（模型不会主动开始/结束文档）")

    print("\n创建模型...")
    model_config = ModelConfig(
        vocab_size=len(tokenizer),
        max_seq_len=max_seq_len,
        d_model=int(model_cfg['d_model']),
        n_heads=int(model_cfg['n_heads']),
        n_layers=int(model_cfg['n_layers']),
        d_ff=int(model_cfg['d_ff']),
        dropout=float(model_cfg['dropout']),
        use_rope=bool(model_cfg['use_rope']),
        tie_weights=bool(model_cfg['tie_weights']),
        use_fp16=train_config.use_fp16,
        use_compile=train_config.use_compile,
        use_gradient_checkpointing=bool(model_cfg['use_gradient_checkpointing']),
    )
    model = DecoderOnlyTransformer(model_config)
    model.set_tokenizer(tokenizer)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n优化特性:")
    print("  ✓ memmap uint16 平铺缓存 (消除单样本 pickle 放大)")
    print("  ✓ 变长微批 / token budget (短序列不再空转)")
    print("  ✓ 分块 cross-entropy")
    print("  ✓ F.rms_norm 融合 RMSNorm")
    print("  ✓ fused AdamW")
    print("  ✓ FlashAttention-2 (V100/sm70)，不可用时回退 SDPA")
    print("  ✓ FP16/BF16 混合精度 (Tensor Cores)")
    print(f"  ✓ 最大序列长度: {max_seq_len}")

    print("\n提示: 按 Ctrl+C 可安全暂停训练并显示菜单")
    print(f"提示: 进度不自动定期保存；在 Ctrl+C 菜单选 1/3/6 写入 "
          f"{os.path.join(checkpoint_dir, 'last.pt')}")
    print("提示: 需要普通检查点（可推理/另起分支）请选菜单 7")
    print(f"提示: 恢复命令: python train.py --config {args.config} --resume")
    print("=" * 60)

    print("\n开始训练...")
    try:
        train_model_optimized(model, train_loader, train_config,
                              save_dir=checkpoint_dir)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        show_interrupt_menu()
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    if not train_config.test_mode and train_config.max_steps == 0:
        final_path = os.path.join(checkpoint_dir, 'final_model.pt')
        get_original_model(model).save(final_path)
    print("\n完成！")


if __name__ == "__main__":
    main()
