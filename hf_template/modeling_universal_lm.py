# -*- coding: utf-8 -*-
"""UniversalLM 因果语言模型（Hugging Face 自定义模型，逐位复现训练代码）。

注意 RoPE 的实现：训练代码先在 **完整 hidden_size** 上计算 inv_freq，
再把 cos/sin 截断到 head_dim 使用。这与标准 Llama 的 per-head RoPE 不同，
因此不能直接映射成 LlamaForCausalLM，必须使用本文件。
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
try:
    from transformers import GenerationMixin
except ImportError:  # 旧版本
    from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:  # 作为包导入（本地目录）
    from .configuration_universal_lm import UniversalLMConfig
except ImportError:  # transformers 动态加载
    from configuration_universal_lm import UniversalLMConfig


# ==================== 基础组件 ====================
class UniversalLMRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class UniversalLMRotaryEmbedding(nn.Module):
    """与训练代码一致：inv_freq 长度 = hidden_size//2。

    这里每次 forward 都按 config 现算 cos/sin，不缓存 buffer —— 避免
    `from_pretrained(low_cpu_mem_usage=True)` 时 non-persistent buffer 被
    `to_empty()` 填成未初始化内存。
    """

    def __init__(self, config: UniversalLMConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.rope_theta = float(config.rope_theta)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        d_model = self.d_model
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, d_model, 2, device=position_ids.device).float() / d_model))
        # position_ids: [B, S] -> freqs: [B, S, d_model/2]
        freqs = torch.einsum('bs,d->bsd', position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, head_dim: int):
    # cos/sin: [B, S, hidden_size] -> 取前 head_dim，与训练代码的截断一致
    cos = cos[..., :head_dim].unsqueeze(1)
    sin = sin[..., :head_dim].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _cache_seq_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, (tuple, list)):
        return int(past_key_values[0][0].shape[-2]) if len(past_key_values) else 0
    if hasattr(past_key_values, 'get_seq_length'):
        try:
            return int(past_key_values.get_seq_length() or 0)
        except TypeError:  # 某些版本需要 layer_idx 参数
            try:
                return int(past_key_values.get_seq_length(0) or 0)
            except Exception:
                return 0
    return 0


def _new_dynamic_cache(config):
    try:
        from transformers.cache_utils import DynamicCache
        try:
            return DynamicCache(config=config)
        except TypeError:
            return DynamicCache()
    except Exception:
        return ()


class UniversalLMAttention(nn.Module):
    def __init__(self, config: UniversalLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.wqkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        use_cache: bool = False,
        is_causal: bool = True,
    ):
        batch_size, seq_len, _ = x.shape

        qkv = self.wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin, self.head_dim)

        if past_key_value is not None:
            if (not isinstance(past_key_value, (tuple, list))
                    and hasattr(past_key_value, 'update')):
                k, v = past_key_value.update(k, v, self.layer_idx)
            else:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)

        present = (k, v) if use_cache else None

        # attention_mask 为 None 时走 is_causal；带 past 的单步解码用全可见
        if attention_mask is None:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=is_causal)
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.wo(attn), present


class UniversalLMFeedForward(nn.Module):
    def __init__(self, config: UniversalLMConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class UniversalLMBlock(nn.Module):
    def __init__(self, config: UniversalLMConfig, layer_idx: int):
        super().__init__()
        self.self_attn = UniversalLMAttention(config, layer_idx)
        self.ff = UniversalLMFeedForward(config)
        self.norm1 = UniversalLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = UniversalLMRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin, attention_mask=None,
                past_key_value=None, use_cache=False, is_causal=True):
        residual = x
        x = self.norm1(x)
        attn_out, present = self.self_attn(
            x, cos, sin, attention_mask, past_key_value, use_cache, is_causal)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x, present


class UniversalLMModel(nn.Module):
    def __init__(self, config: UniversalLMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = UniversalLMRotaryEmbedding(config)
        self.layers = nn.ModuleList([
            UniversalLMBlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm = UniversalLMRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        past_len = _cache_seq_len(past_key_values)

        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids[:, -seq_len:].clamp(min=0)
            else:
                position_ids = torch.arange(
                    past_len, past_len + seq_len,
                    device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        cos, sin = self.rotary_emb(position_ids, x.dtype)

        # 4D attention mask：仅在需要时构造（无 mask 时走 is_causal 快路径）
        kv_len = past_len + seq_len
        attn_mask = None
        if attention_mask is not None:
            min_val = torch.finfo(x.dtype).min
            attn_mask = torch.zeros(
                batch_size, 1, seq_len, kv_len, dtype=x.dtype, device=x.device)
            if seq_len > 1:
                causal = torch.ones(
                    seq_len, kv_len, dtype=torch.bool, device=x.device).triu(
                    diagonal=kv_len - seq_len + 1)
                attn_mask = attn_mask.masked_fill(causal, min_val)
            pad = (1 - attention_mask[:, None, None, :].to(x.dtype)) * min_val
            attn_mask = attn_mask + pad
        elif past_len > 0 and seq_len > 1:
            # 带 KV cache 的多 token 块：SDPA 的 is_causal 对非方阵是左上对齐，
            # 必须显式构造右下对齐的 causal mask
            min_val = torch.finfo(x.dtype).min
            causal = torch.ones(
                seq_len, kv_len, dtype=torch.bool, device=x.device).triu(
                diagonal=kv_len - seq_len + 1)
            attn_mask = torch.zeros(
                batch_size, 1, seq_len, kv_len, dtype=x.dtype, device=x.device)
            attn_mask = attn_mask.masked_fill(causal, min_val)

        is_causal = attn_mask is None and past_len == 0

        is_dynamic_cache = (
            past_key_values is not None
            and not isinstance(past_key_values, (tuple, list)))
        next_cache = () if not is_dynamic_cache else past_key_values

        for i, layer in enumerate(self.layers):
            if is_dynamic_cache:
                past_layer = past_key_values
            elif isinstance(past_key_values, (tuple, list)) and i < len(past_key_values):
                past_layer = past_key_values[i]
            else:
                past_layer = None
            x, present = layer(
                x, cos, sin, attn_mask, past_layer, use_cache, is_causal)
            if use_cache and not is_dynamic_cache:
                next_cache = next_cache + (present,)

        x = self.norm(x)
        return x, (next_cache if use_cache else None)


class UniversalLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = UniversalLMConfig
    base_model_prefix = 'model'
    _tied_weights_keys = {'lm_head.weight': 'model.embedding.weight'}
    supports_gradient_checkpointing = False

    def __init__(self, config: UniversalLMConfig):
        super().__init__(config)
        self.model = UniversalLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embedding.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def _init_weights(self, module):
        std = getattr(self.config, 'initializer_range', 0.02)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if use_cache is None:
            use_cache = getattr(self.config, 'use_cache', True)
        if inputs_embeds is not None:
            raise NotImplementedError('inputs_embeds 暂不支持，请传 input_ids')

        hidden, next_cache = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None and _cache_seq_len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache', True),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return None
        if isinstance(past_key_values, (tuple, list)):
            return tuple(
                (k.index_select(0, beam_idx.to(k.device)),
                 v.index_select(0, beam_idx.to(v.device)))
                for k, v in past_key_values)
        if hasattr(past_key_values, 'batch_select_indices'):
            return past_key_values.batch_select_indices(beam_idx)
        if hasattr(past_key_values, 'reorder_cache'):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        return past_key_values
