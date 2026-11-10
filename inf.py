#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer 推理脚本（完全独立单文件，不依赖 train.py）

用法:
    python inf.py --model ./checkpoints/last.pt --tokenizer ./tokenizer.json --prompt "今天"
    python inf.py --model ./checkpoints/last.pt --tokenizer ./tokenizer.json --interactive
    python inf.py --model ./checkpoints/last.pt --tokenizer ./tokenizer.json --batch_file prompts.txt
    python inf.py ... --prompt "今天" --no_stream     # 关闭流式，一次性输出
    python inf.py ... --prompt "今天" --repetition_penalty 1.15 --no_repeat_ngram_size 3
                                                                          # 重复惩罚 / 禁止重复 n-gram
    python inf.py --model ./checkpoints/last.pt --tokenizer ./tokenizer.json \
        --test data/chinese.jsonl --test_limit 1000 --device cuda --fp16   # 评分
    python inf.py ... --test data/test_samples/mixed.jsonl --test_detail  # 逐文档明细
    python inf.py ... --test a.jsonl,b.jsonl,c.jsonl --test_detail        # 按文件分类汇总
    python inf.py ... --test ... --test_bos/--no-test_bos --test_eos/--no-test_eos
                                                                          # 覆盖 checkpoint 的 BOS/EOS 口径
"""

import os
import sys
import time
import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


# ==================== 特殊 token（与 train.py 一致） ====================
SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']


# ==================== 配置类（与 train.py 一致） ====================
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


# ==================== 模型组件（与 train.py 一致） ====================
class FusedRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
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
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,
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

    def forward_hidden(self, x, mask=None):
        """返回最后一层 norm 后的 hidden state（评分时分块接 fc_out，避免整块 logits 爆显存）。"""
        batch_size, seq_len = x.shape
        x = self.embedding(x)

        cos, sin = None, None
        if self.config.use_rope:
            cos, sin = self.rope(seq_len)
            cos = cos.to(x.dtype)
            sin = sin.to(x.dtype)

        for layer in self.layers:
            x = layer(x, mask=mask, cos=cos, sin=sin)

        return self.norm(x)

    def forward(self, x, mask=None):
        return self.fc_out(self.forward_hidden(x, mask))

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


# ==================== 分词器封装（与 train.py 一致） ====================
class BPETokenizer:
    def __init__(self, vocab_size: int = 66536):
        self.vocab_size = vocab_size
        self.special_tokens = SPECIAL_TOKENS
        self.tokenizer = None
        # 编码正文用的副本：不含 <bos>/<eos> added token，避免文本里字面出现的
        # "<bos>"/"<eos>" 被识别成特殊 token id。decode 仍用原始分词器。
        self._plain_tokenizer = None
        self.special_ids = {}

    def _build_plain_tokenizer(self):
        self._plain_tokenizer = None
        if self.tokenizer is None:
            return
        try:
            cfg = json.loads(self.tokenizer.to_str())
            added = cfg.get('added_tokens', [])
            plain = [t for t in added
                     if t.get('content') not in ('<bos>', '<eos>')]
            if len(plain) == len(added):
                self._plain_tokenizer = self.tokenizer
                return
            cfg['added_tokens'] = plain
            self._plain_tokenizer = Tokenizer.from_str(
                json.dumps(cfg, ensure_ascii=False))
        except Exception as e:
            print(f"⚠️ 构建正文编码器失败({e})，回退原分词器："
                  f"文本内的 <bos>/<eos> 仍会被当作特殊 token")
            self._plain_tokenizer = self.tokenizer

    def encode(self, text: str) -> List[int]:
        return (self._plain_tokenizer or self.tokenizer).encode(text).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """批量编码（tokenizers 的 Rust 多线程实现），评分时用。"""
        if self.tokenizer is None:
            raise RuntimeError("分词器未加载")
        tok = self._plain_tokenizer or self.tokenizer
        return [e.ids for e in tok.encode_batch(texts)]

    def decode(self, ids: List[int]) -> str:
        # 与旧版“特殊 token 在 decode 时被跳过”的行为保持一致，避免生成结果
        # 末尾出现字面 "<eos>"（移除 added token 后原始 decode 会保留它）。
        skip = set(self.special_ids.values())
        if skip:
            ids = [int(i) for i in ids if int(i) not in skip]
        return self.tokenizer.decode(ids)

    @property
    def pad_token_id(self):
        return self.special_ids.get('<pad>', 0)

    @property
    def bos_token_id(self):
        return self.special_ids.get('<bos>', 2)

    @property
    def eos_token_id(self):
        return self.special_ids.get('<eos>', 3)

    def __len__(self):
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        for token in self.special_tokens:
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                self.special_ids[token] = token_id
        self._build_plain_tokenizer()
        print(f"✓ 分词器加载成功: {len(self)} 词表"
              f"（文本内字面 <bos>/<eos> 按普通 BPE 编码）")


# ==================== 采样辅助 ====================
def _apply_sampling(
    next_logits: torch.Tensor,
    generated_ids: List[int],
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.15,
    repetition_window: int = 512,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """对最后一步的 logits 做重复惩罚 / 温度 / top-k / top-p，返回概率分布。

    - repetition_penalty: CTRL 风格的重复惩罚。>1 会降低已出现 token 的概率，
      1.0 表示关闭。
    - repetition_window: 只惩罚最近 N 个 token；0 表示惩罚全部历史。
    - no_repeat_ngram_size: 禁止重复的 n-gram（0=关闭，常用 3）。这是硬屏蔽：
      如果下一个 token 会和历史中相同的 n-1 前缀构成重复 n-gram，直接置 -inf。
    """
    original = next_logits
    next_logits = next_logits.clone()

    # 1) 重复惩罚
    if repetition_penalty != 1.0 and generated_ids:
        recent = (generated_ids[-repetition_window:]
                  if repetition_window and repetition_window > 0
                  else generated_ids)
        vocab = next_logits.size(-1)
        for token_id in set(recent):
            if 0 <= token_id < vocab:
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty

    # 2) 禁止重复 n-gram（硬屏蔽）
    n = int(no_repeat_ngram_size or 0)
    if n >= 1 and len(generated_ids) >= n:
        prefix = tuple(generated_ids[-(n - 1):]) if n > 1 else ()
        banned = set()
        for i in range(len(generated_ids) - n + 1):
            if tuple(generated_ids[i:i + n - 1]) == prefix:
                banned.add(generated_ids[i + n - 1])
        banned = [t for t in banned if 0 <= t < next_logits.size(-1)]
        if banned:
            next_logits[0, banned] = float('-inf')

    # 3) 温度
    if temperature > 0:
        next_logits = next_logits / temperature

    # 4) top-k
    if top_k > 0:
        k = min(top_k, next_logits.size(-1))
        top_logits, top_indices = torch.topk(next_logits, k)
        mask = torch.full_like(next_logits, float('-inf'))
        mask.scatter_(-1, top_indices, top_logits)
        next_logits = mask

    # 5) top-p
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        sorted_mask = cum > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        next_logits = next_logits.masked_fill(remove, float('-inf'))

    probs = F.softmax(next_logits, dim=-1)
    # 极端情况下（例如 no-repeat n-gram 把所有候选都屏蔽了）回退到原始分布
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        probs = F.softmax(original, dim=-1)
    return probs


# ==================== 流式生成核心 ====================
@torch.no_grad()
def generate_stream(
    model: DecoderOnlyTransformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    repetition_window: int = 512,
    no_repeat_ngram_size: int = 0,
    device: torch.device = torch.device('cpu'),
    use_amp: bool = False,
    include_prompt: bool = False,
    add_bos: Optional[bool] = None,
) -> Iterator[str]:
    """
    流式生成器：每产生一个（或一批）新 token，就 yield 增量文本片段。
    - include_prompt=True 时，先 yield 一次 prompt 原文。
    - add_bos=None 时跟随 checkpoint（训练时是否加了 BOS）。
    - 使用 token 缓冲区 + 解码再编码的"前缀比对"技巧，避免 BPE 边界产生乱码。
    """
    model.eval()

    if add_bos is None:
        add_bos = bool(getattr(model, 'bos_supervised', False))
    prompt_ids = list(tokenizer.encode(prompt))
    input_ids = ([tokenizer.bos_token_id] if add_bos else []) + prompt_ids
    if not input_ids:                    # 空 prompt 且不加 BOS 时避免 0 长度输入
        input_ids = [tokenizer.bos_token_id]

    max_seq_len = model.config.max_seq_len
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_ids = list(input_ids)

    amp_dtype = torch.float16 if (use_amp and torch.cuda.is_available()) else None

    if include_prompt:
        yield prompt

    # 已输出的文本（用于计算增量）
    emitted_text = ""

    for _ in range(max_new_tokens):
        if input_tensor.shape[1] > max_seq_len:
            input_tensor = input_tensor[:, -max_seq_len:]

        if amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(input_tensor)
        else:
            logits = model(input_tensor)

        next_logits = logits[:, -1, :].float()

        probs = _apply_sampling(
            next_logits, generated_ids,
            temperature, top_k, top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        if temperature <= 0:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        generated_ids.append(token_id)
        input_tensor = torch.cat([input_tensor, next_token], dim=1)

        if token_id == tokenizer.eos_token_id:
            break

        # ---- 解码 & 计算增量 ----
        # 直接 decode 整个序列可能因为结尾 token 尚未完整而出现乱码，
        # 因此采用「解码 -> 与已输出文本比对」的方式：
        # 把新 token 的解码结果追加到缓冲区，再整体 decode 后取增量。
        # 这里用一个更稳的做法：对最新一小段 token 做 decode，找到增量。
        full_text = tokenizer.decode(generated_ids)

        # 只有当 full_text 以 emitted_text 为前缀时，才认为增量是稳定的。
        # 否则（BPE 出现边界不稳定），先不输出，等下一个 token。
        if full_text.startswith(emitted_text):
            delta = full_text[len(emitted_text):]
            if delta:
                emitted_text = full_text
                yield delta
        else:
            # 边界不稳定：先回退，等下一次 decode 再看
            # 实际上 emitted_text 不变，full_text 会随着 token 增加逐渐稳定
            pass

    # 收尾：把剩余未输出的部分吐出来（如果有）
    full_text = tokenizer.decode(generated_ids)
    if full_text.startswith(emitted_text) and len(full_text) > len(emitted_text):
        yield full_text[len(emitted_text):]


# ==================== 非流式生成（保留原接口，用于批量） ====================
@torch.no_grad()
def generate(
    model: DecoderOnlyTransformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    repetition_window: int = 512,
    no_repeat_ngram_size: int = 0,
    device: torch.device = torch.device('cpu'),
    use_amp: bool = False,
    add_bos: Optional[bool] = None,
) -> str:
    """自回归生成，返回完整文本（包含 prompt）。"""
    model.eval()

    if add_bos is None:
        add_bos = bool(getattr(model, 'bos_supervised', False))
    prompt_ids = list(tokenizer.encode(prompt))
    input_ids = ([tokenizer.bos_token_id] if add_bos else []) + prompt_ids
    if not input_ids:
        input_ids = [tokenizer.bos_token_id]

    max_seq_len = model.config.max_seq_len
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_ids = list(input_ids)

    amp_dtype = torch.float16 if (use_amp and torch.cuda.is_available()) else None

    for _ in range(max_new_tokens):
        if input_tensor.shape[1] > max_seq_len:
            input_tensor = input_tensor[:, -max_seq_len:]

        if amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(input_tensor)
        else:
            logits = model(input_tensor)

        next_logits = logits[:, -1, :].float()

        probs = _apply_sampling(
            next_logits, generated_ids,
            temperature, top_k, top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        if temperature <= 0:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        generated_ids.append(next_token.item())
        input_tensor = torch.cat([input_tensor, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids)


# ==================== 加载 ====================
def load_model(model_path: str, tokenizer_path: str, device: torch.device
               ) -> Tuple[DecoderOnlyTransformer, BPETokenizer]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"分词器文件不存在: {tokenizer_path}")

    print(f"加载分词器: {tokenizer_path}")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    print(f"  - 词表大小: {len(tokenizer)}")
    print(f"  - BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, "
          f"PAD={tokenizer.pad_token_id}")

    print(f"\n加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # 训练端是否在样本前后监督了 BOS/EOS。新 checkpoint 顶层有标记；
    # 旧 checkpoint 没有这些字段时按 False（未训练）处理，并提示用户。
    eos_supervised = None
    bos_supervised = None
    if isinstance(checkpoint, dict):
        if 'eos_supervised' in checkpoint:
            eos_supervised = bool(checkpoint['eos_supervised'])
        if 'bos_supervised' in checkpoint:
            bos_supervised = bool(checkpoint['bos_supervised'])
        _fp = ((checkpoint.get('training') or {}).get('data_fingerprint') or {})
        if eos_supervised is None and 'append_eos' in _fp:
            eos_supervised = bool(_fp['append_eos'])
        if bos_supervised is None and 'prepend_bos' in _fp:
            bos_supervised = bool(_fp['prepend_bos'])

    if isinstance(checkpoint, dict) and checkpoint.get('config') is not None:
        cfg = checkpoint['config']
        if isinstance(cfg, ModelConfig):
            config = cfg
        elif isinstance(cfg, dict):
            config = ModelConfig(**{
                k: v for k, v in cfg.items()
                if k in ModelConfig.__dataclass_fields__
            })
        else:
            config = ModelConfig()
        print("  - 从 checkpoint 恢复 config")
    else:
        print("  ⚠️ checkpoint 中无 config，使用 ModelConfig 默认值")
        config = ModelConfig()

    if config.vocab_size != len(tokenizer):
        print(f"  ⚠️ config.vocab_size={config.vocab_size} 与 tokenizer={len(tokenizer)} "
              f"不一致，以 tokenizer 为准")
        config.vocab_size = len(tokenizer)

    config.use_gradient_checkpointing = False

    model = DecoderOnlyTransformer(config)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  ⚠️ 缺失权重: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  ⚠️ 多余权重: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    model = model.to(device)
    model.eval()
    model.set_tokenizer(tokenizer)
    model.eos_supervised = (bool(eos_supervised) if eos_supervised is not None
                            else False)
    model.eos_supervised_known = eos_supervised is not None
    model.bos_supervised = (bool(bos_supervised) if bos_supervised is not None
                            else False)
    model.bos_supervised_known = bos_supervised is not None

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - 参数量: {total_params:,}")
    print(f"  - 设备: {device}")
    if eos_supervised is None or bos_supervised is None:
        print("  - BOS/EOS 监督: 未知（旧 checkpoint 无标记，按“关”处理；"
              "可用 --test_bos/--test_eos 强制开启）")
    else:
        print(f"  - BOS 监督: {'开' if bos_supervised else '关'}  "
              f"EOS 监督: {'开' if eos_supervised else '关'}")

    return model, tokenizer


# ==================== CLI ====================
def parse_args():
    p = argparse.ArgumentParser(description="Transformer 推理脚本（独立单文件，支持流式）")
    p.add_argument('--model', type=str, default='./checkpoints/last.pt',
                   help='模型权重路径 (.pt)')
    p.add_argument('--tokenizer', type=str, default='./tokenizer.json',
                   help='分词器路径 (tokenizer.json)')
    p.add_argument('--prompt', type=str, default=None,
                   help='单次生成提示词')
    p.add_argument('--interactive', action='store_true',
                   help='进入交互式对话模式')
    p.add_argument('--batch_file', type=str, default=None,
                   help='批量提示词文件（每行一个提示词）')
    p.add_argument('--max_new_tokens', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.8,
                   help='温度（<=0 为贪心解码）')
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--repetition_penalty', type=float, default=1.15,
                   help='重复惩罚系数，>1 抑制重复；1.0 关闭（默认 1.15）')
    p.add_argument('--repetition_window', type=int, default=512,
                   help='重复惩罚回看窗口（token 数），0 表示全部历史（默认 512）')
    p.add_argument('--no_repeat_ngram_size', type=int, default=0,
                   help='禁止重复 n-gram 的长度，0 关闭，常用 3（默认 0）')
    p.add_argument('--device', type=str, default='cpu',
                   help='设备: auto/cuda/cpu')
    p.add_argument('--fp16', action='store_true',
                   help='使用 FP16 推理（仅 CUDA 有效）')
    _bos = p.add_mutually_exclusive_group()
    _bos.add_argument('--add_bos', '--add-bos', dest='add_bos',
                      action='store_true', default=None,
                      help='生成时前置 BOS（默认跟随 checkpoint 的 BOS 设置）')
    _bos.add_argument('--no_add_bos', '--no-add-bos', dest='add_bos',
                      action='store_false',
                      help='生成时不前置 BOS')
    p.add_argument('--seed', type=int, default=42)
    # ---- 新增：流式控制 ----
    p.add_argument('--no_stream', action='store_true',
                   help='关闭流式输出，一次性打印完整结果')
    p.add_argument('--show_prompt', action='store_true',
                   help='流式模式下先回显 prompt')
    # ---- 新增：评分 ----
    p.add_argument('--test', nargs='?', const='', default=None,
                   metavar='FILE[,FILE...]',
                   help='评分模式：语料文件（可用逗号分隔多个文件，按文件分类汇总）')
    p.add_argument('--test_limit', type=int, default=1000,
                   help='评分最多读取的文档数（0 = 全部；多文件时每个文件各自生效）')
    p.add_argument('--test_seq_len', type=int, default=0,
                   help='评分窗口长度（0 = 使用模型 max_seq_len）')
    p.add_argument('--test_batch', type=int, default=4,
                   help='评分 batch size（默认 4）')
    p.add_argument('--test_chunk_rows', type=int, default=1024,
                   help='fc_out+CE 每次处理的行数（默认 1024，控制显存）')
    p.add_argument('--test_detail', action='store_true',
                   help='输出逐文档明细（loss/ppl/acc）以及最好/最差样本')
    p.add_argument('--test_topn', type=int, default=5,
                   help='明细模式下最好/最差各展示多少篇（默认 5，0 = 不展示）')
    _eos = p.add_mutually_exclusive_group()
    _eos.add_argument('--test_eos', '--test-eos', dest='test_eos',
                      action='store_true', default=None,
                      help='评分时在每篇文档末尾追加 EOS（默认跟随 checkpoint 的 EOS 设置）')
    _eos.add_argument('--no-test_eos', '--no-test-eos', dest='test_eos',
                      action='store_false',
                      help='评分时不追加 EOS')
    _tbos = p.add_mutually_exclusive_group()
    _tbos.add_argument('--test_bos', '--test-bos', dest='test_bos',
                       action='store_true', default=None,
                       help='评分时在每篇文档开头加 BOS（默认跟随 checkpoint 的 BOS 设置）')
    _tbos.add_argument('--no-test_bos', '--no-test-bos', dest='test_bos',
                       action='store_false',
                       help='评分时不加 BOS')
    return p.parse_args()


def resolve_device(s: str) -> torch.device:
    if s == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(s)


def stream_and_print(model, tokenizer, prompt, args, device,
                     include_prompt: bool = False) -> str:
    """流式生成并实时打印，返回完整生成文本。"""
    t0 = time.time()
    pieces: List[str] = []
    first_token_time = None

    if include_prompt:
        # 回显 prompt（不参与首 token 计时）
        print(prompt, end='', flush=True)

    for delta in generate_stream(
        model, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        device=device,
        use_amp=args.fp16,
        add_bos=getattr(args, 'add_bos', None),
        include_prompt=False,  # prompt 由外层负责回显
    ):
        if first_token_time is None:
            first_token_time = time.time() - t0
        pieces.append(delta)
        print(delta, end='', flush=True)

    elapsed = time.time() - t0
    print()  # 换行
    if first_token_time is not None:
        print(f"\n[首 token {first_token_time:.2f}s | 总耗时 {elapsed:.2f}s]")
    else:
        print(f"\n[总耗时 {elapsed:.2f}s]")
    return ''.join(pieces)


def interactive_mode(model, tokenizer, args, device):
    print("\n" + "=" * 60)
    print("交互式生成模式（输入 'exit' / 'quit' 退出）")
    if args.no_stream:
        print("模式: 非流式")
    else:
        print("模式: 流式（逐 token 输出）")
    print("=" * 60)
    while True:
        try:
            prompt = input("\n>>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            break
        if prompt.lower() in ('exit', 'quit', 'q'):
            break
        if not prompt:
            continue

        if args.no_stream:
            t0 = time.time()
            output = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                device=device,
                use_amp=args.fp16,
                add_bos=getattr(args, 'add_bos', None),
            )
            elapsed = time.time() - t0
            print(f"\n{output}")
            print(f"\n[耗时 {elapsed:.2f}s]")
        else:
            print()  # 输出前空一行
            stream_and_print(model, tokenizer, prompt, args, device,
                             include_prompt=False)


# ==================== 评分（--test） ====================
def iter_test_texts(path: str, limit: int):
    """逐条产出评分文本：.jsonl 取 text/content 字段，其他文件按行。

    limit <= 0 表示读取全部。只统计有效文本，空行/坏行/空字段会被跳过。
    """
    use_json = os.path.splitext(path)[1].lower() in ('.jsonl', '.json', '.ndjson')
    n = 0
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            if not raw.strip():
                continue
            if use_json:
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                text = obj.get('text', obj.get('content', ''))
            else:
                text = raw.strip()
            if not text or not isinstance(text, str):
                continue
            yield text
            n += 1
            if limit > 0 and n >= limit:
                return


def build_scoring_sequence(ids, use_bos: bool, bos_id: int,
                           use_eos: bool, eos_id: int):
    """与训练一致：可选地在文档前后加 BOS/EOS。"""
    seq = list(ids)
    if use_bos:
        seq.insert(0, bos_id)
    if use_eos:
        seq.append(eos_id)
    return seq


def split_scoring_windows(ids, seq_len: int):
    """把一条 token 序列切成 input 宽度不超过 seq_len 的窗口。

    窗口 w 用 w[:-1] 作输入、w[1:] 作标签，因此长度 <= seq_len 的序列只需
    一个窗口；长序列按 seq_len 步长切分，相邻窗口重叠 1 个 token 作为上下文，
    保证每个 token（含末尾 EOS）恰好被预测一次。
    """
    if len(ids) <= seq_len:
        return [list(ids)]
    wins = []
    for start in range(0, len(ids), seq_len):
        w = ids[start:start + seq_len + 1]
        if len(w) >= 2:  # 末尾残余的 1 token 已由前一个窗口预测过
            wins.append(w)
    return wins


def score_windows(model, windows, pad_id, device, batch_size, chunk_rows, amp):
    """对窗口列表逐窗口打分，返回与 windows 顺序对齐的三个列表。

    (losses, tokens, correct)
    每个窗口形如 ids[0..L]，input=ids[:-1]、label=ids[1:]，因此窗口内每个
    非首 token 都被恰好预测一次；长文档按 seq_len 步长切窗，块间重叠 1 个
    token，保证既不重复也不遗漏。
    """
    n = len(windows)
    losses = [0.0] * n
    tokens = [0] * n
    correct = [0] * n
    if n == 0:
        return losses, tokens, correct

    # 按长度排序后组 batch，减少 padding 浪费（右侧 padding 不影响因果注意力）
    order = sorted(range(n), key=lambda i: len(windows[i]))

    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        group = [windows[i] for i in idxs]
        t = max(len(w) for w in group) - 1
        if t < 1:
            continue

        input_ids = torch.full((len(group), t), pad_id, dtype=torch.long)
        labels = torch.full((len(group), t), -100, dtype=torch.long)
        for j, w in enumerate(group):
            length = len(w) - 1
            input_ids[j, :length] = torch.tensor(w[:-1], dtype=torch.long)
            labels[j, :length] = torch.tensor(w[1:], dtype=torch.long)

        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16, enabled=amp):
                hidden = model.forward_hidden(input_ids)

            flat_h = hidden.reshape(-1, hidden.shape[-1])
            flat_y = labels.reshape(-1)

            # logits [B, T, vocab] 太大，按行分块做 fc_out + cross_entropy
            for s in range(0, flat_y.numel(), chunk_rows):
                y = flat_y[s:s + chunk_rows]
                if int((y != -100).sum()) == 0:
                    continue
                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16, enabled=amp):
                    logits = model.fc_out(flat_h[s:s + chunk_rows])
                    per_pos = F.cross_entropy(
                        logits, y, ignore_index=-100, reduction='none')
                pred = logits.argmax(dim=-1)

                # 把分块里的行按 batch 元素归并回窗口
                for local, orig in enumerate(idxs):
                    lo = max(s, local * t)
                    hi = min(s + chunk_rows, (local + 1) * t)
                    if hi <= lo:
                        continue
                    a, b = lo - s, hi - s
                    yb = y[a:b]
                    keep = yb != -100
                    nb = int(keep.sum())
                    if nb == 0:
                        continue
                    losses[orig] += float(per_pos[a:b][keep].sum())
                    tokens[orig] += nb
                    correct[orig] += int((pred[a:b][keep] == yb[keep]).sum())

    return losses, tokens, correct


def _fmt_ppl(avg_loss: float) -> str:
    return f"{math.exp(avg_loss):,.2f}" if avg_loss < 700 else "inf"


def _summarize(records):
    pred = sum(r['pred_tokens'] for r in records)
    if pred == 0:
        return None
    return {
        'docs': len(records),
        'windows': sum(r['windows'] for r in records),
        'tokens': sum(r['tokens'] for r in records),
        'pred': pred,
        'loss': sum(r['loss'] for r in records),
        'correct': sum(r['correct'] for r in records),
    }


def _print_metrics(s, skipped, elapsed):
    avg = s['loss'] / s['pred']
    acc = s['correct'] / s['pred']
    bits = avg / math.log(2)
    print(f"  有效文档:      {s['docs']:,}（跳过 {skipped:,} 篇空/过短文档）")
    print(f"  窗口数:        {s['windows']:,}")
    print(f"  预测 token:    {s['pred']:,}")
    print(f"  平均 loss:     {avg:.6f}")
    print(f"  perplexity:    {_fmt_ppl(avg)}")
    print(f"  bits/token:    {bits:.6f}")
    print(f"  next-token 准确率: {acc:.4%}")
    print(f"  耗时:          {elapsed:.2f}s "
          f"({s['pred'] / max(elapsed, 1e-9):,.1f} token/s)")


def _print_detail(records, topn):
    print("\n  逐文档明细（loss/acc 只统计该文档被预测的 token）:")
    print("    #     tok  win      loss        ppl      acc   预览")
    for i, r in enumerate(records, 1):
        n = r['pred_tokens']
        avg = r['loss'] / n
        acc = r['correct'] / n
        print(f"  {i:>3}  {r['tokens']:>5}  {r['windows']:>3}  {avg:9.4f}  "
              f"{_fmt_ppl(avg):>9}  {acc:7.2%}   {r['preview']}")

    if topn > 0 and len(records) > 1:
        ranked = sorted(records, key=lambda r: r['loss'] / r['pred_tokens'])
        best = ranked[:topn]
        worst = ranked[-topn:][::-1]
        print(f"\n  最好 {len(best)} 篇（loss 最低）:")
        for r in best:
            n = r['pred_tokens']
            print(f"    loss={r['loss'] / n:.4f}  acc={r['correct'] / n:.2%}  "
                  f"tok={r['tokens']}  {r['preview']}")
        print(f"\n  最差 {len(worst)} 篇（loss 最高）:")
        for r in worst:
            n = r['pred_tokens']
            print(f"    loss={r['loss'] / n:.4f}  acc={r['correct'] / n:.2%}  "
                  f"tok={r['tokens']}  {r['preview']}")


def evaluate_model(model, tokenizer, args, device):
    paths = [p.strip() for p in (args.test or '').split(',') if p.strip()]
    if not paths:
        print("❌ --test 需要语料文件路径，例如：")
        print("   python inf.py --model ./checkpoints/last.pt --tokenizer ./tokenizer.json \\")
        print("       --test data/test_samples/mixed.jsonl --test_detail --device cuda --fp16")
        return
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"❌ 评分语料不存在: {p}")
        return

    max_seq_len = model.config.max_seq_len
    seq_len = args.test_seq_len or max_seq_len
    if seq_len > max_seq_len:
        print(f"⚠️ --test_seq_len={seq_len} 超过模型上限，已截断为 {max_seq_len}")
        seq_len = max_seq_len
    if seq_len < 2:
        print("❌ --test_seq_len 至少为 2")
        return

    batch_size = max(1, args.test_batch)
    chunk_rows = max(1, args.test_chunk_rows)
    topn = max(0, args.test_topn)
    amp = bool(args.fp16 and device.type == 'cuda')
    if args.fp16 and device.type != 'cuda':
        print("⚠️ --fp16 仅对 CUDA 有效，本次按 FP32 评分")

    print("\n" + "=" * 66)
    print("评分模式")
    print("=" * 66)
    print(f"语料: {len(paths)} 个文件" + ("（逗号分隔）" if len(paths) > 1 else ""))
    for p in paths:
        print(f"  - {p}")
    print(f"文档上限: {'全部' if args.test_limit <= 0 else f'每文件 {args.test_limit:,}'}")
    print(f"窗口长度: {seq_len} token"
          + ("（模型 max_seq_len）" if args.test_seq_len == 0 else ""))
    print(f"batch={batch_size}  CE 分块={chunk_rows}  "
          f"精度={'FP16' if amp else 'FP32'}")
    print(f"逐文档明细: {'开' if args.test_detail else '关'}"
          + (f"（最好/最差各 {topn} 篇）" if args.test_detail else ""))
    use_eos = (bool(args.test_eos) if getattr(args, 'test_eos', None) is not None
               else bool(getattr(model, 'eos_supervised', False)))
    use_bos = (bool(args.test_bos) if getattr(args, 'test_bos', None) is not None
               else bool(getattr(model, 'bos_supervised', False)))
    eos_src = '命令行' if getattr(args, 'test_eos', None) is not None else (
        'checkpoint' if getattr(model, 'eos_supervised_known', False) else '默认(旧ckpt)')
    bos_src = '命令行' if getattr(args, 'test_bos', None) is not None else (
        'checkpoint' if getattr(model, 'bos_supervised_known', False) else '默认(旧ckpt)')
    print(f"BOS/EOS 口径: {'BOS + ' if use_bos else ''}正文"
          f"{' + EOS' if use_eos else ''}"
          f"（BOS={bos_src}, EOS={eos_src}）")
    if device.type == 'cpu':
        print("提示: 在 V100 上建议加 --device cuda --fp16")

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    model.eval()
    DOC_CHUNK = 128

    def collect_file(path):
        records = []
        skipped = 0
        windows_total = 0
        min_len = 1 if (use_bos or use_eos) else 2

        def process(texts):
            nonlocal skipped, windows_total
            encodings = tokenizer.encode_batch(texts)
            flat_windows = []
            metas = []
            for text, ids in zip(texts, encodings):
                if len(ids) < min_len:
                    skipped += 1
                    continue
                # 与训练一致：BOS + 正文 + EOS。窗口切分对整条序列进行，
                # 因此 BOS 只作为第一窗口的上下文，EOS 作为最后一个 target。
                seq = build_scoring_sequence(ids, use_bos, bos_id,
                                             use_eos, eos_id)
                wins = split_scoring_windows(seq, seq_len)
                preview = ' '.join(text.split())
                if len(preview) > 52:
                    preview = preview[:52] + '…'
                metas.append((preview, len(ids), len(wins), len(flat_windows)))
                flat_windows.extend(wins)

            windows_total += len(flat_windows)
            losses, toks, corrs = score_windows(
                model, flat_windows, pad_id, device, batch_size, chunk_rows, amp)

            for preview, ntok, nwin, w0 in metas:
                w1 = w0 + nwin
                records.append({
                    'preview': preview,
                    'tokens': ntok,
                    'windows': nwin,
                    'loss': sum(losses[w0:w1]),
                    'pred_tokens': sum(toks[w0:w1]),
                    'correct': sum(corrs[w0:w1]),
                })

        buffer = []
        for text in iter_test_texts(path, args.test_limit):
            buffer.append(text)
            if len(buffer) >= DOC_CHUNK:
                process(buffer)
                buffer = []
        if buffer:
            process(buffer)
        return records, skipped

    results = []
    for path in paths:
        t0 = time.time()
        records, skipped = collect_file(path)
        results.append((path, records, skipped, time.time() - t0))

    if len(results) == 1:
        path, records, skipped, elapsed = results[0]
        s = _summarize(records)
        print(f"\n{'-' * 66}")
        if s is None:
            print("❌ 没有可评分的 token（语料为空或全部被跳过）")
            return
        print("评分结果")
        _print_metrics(s, skipped, elapsed)
        if args.test_detail:
            _print_detail(records, topn)
        print("=" * 66)
        return

    total_records = []
    total_skipped = 0
    total_elapsed = 0.0
    for i, (path, records, skipped, elapsed) in enumerate(results, 1):
        total_records.extend(records)
        total_skipped += skipped
        total_elapsed += elapsed
        print(f"\n{'-' * 66}")
        print(f"[{i}/{len(results)}] {path}")
        s = _summarize(records)
        if s is None:
            print("  ⚠️ 没有可评分的 token（跳过或全部为空）")
            continue
        _print_metrics(s, skipped, elapsed)
        if args.test_detail:
            _print_detail(records, topn)

    s = _summarize(total_records)
    print(f"\n{'=' * 66}")
    if s is None:
        print("❌ 所有文件都没有可评分的 token")
        return
    print(f"总汇总（{len(results)} 个文件）")
    _print_metrics(s, total_skipped, total_elapsed)
    print("=" * 66)


def main():
    args = parse_args()

    # ---- 评分参数先做快速校验，避免白加载 1.7GB checkpoint ----
    if args.test is not None:
        paths = [p.strip() for p in args.test.split(',') if p.strip()]
        if not paths:
            print("❌ --test 需要语料文件路径，例如：")
            print("   python inf.py --model ./checkpoints/last.pt --tokenizer ./tokenizer.json \\")
            print("       --test data/test_samples/mixed.jsonl --test_detail --device cuda --fp16")
            return
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            for p in missing:
                print(f"❌ 评分语料不存在: {p}")
            return

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    model, tokenizer = load_model(args.model, args.tokenizer, device)

    # ---- 评分模式（优先于生成） ----
    if args.test is not None:
        evaluate_model(model, tokenizer, args, device)
        return

    # ---- 批量文件（默认非流式，避免刷屏；如需流式可自行改） ----
    if args.batch_file:
        if not os.path.exists(args.batch_file):
            print(f"❌ 批量提示文件不存在: {args.batch_file}")
            sys.exit(1)
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"\n批量生成 {len(prompts)} 条提示词...")
        for i, prompt in enumerate(prompts):
            t0 = time.time()
            output = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                device=device,
                use_amp=args.fp16,
                add_bos=getattr(args, 'add_bos', None),
            )
            elapsed = time.time() - t0
            print(f"\n[{i+1}/{len(prompts)}] 提示: {prompt}")
            print(f"生成: {output}")
            print(f"[耗时 {elapsed:.2f}s]")
        return

    # ---- 单次提示 ----
    if args.prompt is not None:
        if args.no_stream:
            t0 = time.time()
            output = generate(
                model, tokenizer, args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                device=device,
                use_amp=args.fp16,
                add_bos=getattr(args, 'add_bos', None),
            )
            elapsed = time.time() - t0
            print(f"\n提示: {args.prompt}")
            print(f"生成: {output}")
            print(f"\n[耗时 {elapsed:.2f}s]")
        else:
            print(f"提示: {args.prompt}")
            print("生成: ", end='', flush=True)
            stream_and_print(model, tokenizer, args.prompt, args, device,
                             include_prompt=False)
        return

    # ---- 默认进入交互模式 ----
    interactive_mode(model, tokenizer, args, device)


if __name__ == '__main__':
    main()