#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把本项目训练出的 checkpoint 打包成本地 Hugging Face 模型目录。

默认导出为**自定义架构**（UniversalLMForCausalLM），逐位复现训练代码，包括本项目
RoPE 的特殊实现（在完整 hidden_size 上算 inv_freq，再截断到 head_dim）。
加载时需要 `trust_remote_code=True`。

也可以 `--format llama` 导出成标准 LlamaForCausalLM（无需 trust_remote_code），
但注意：标准 Llama 的 RoPE 与本项目训练代码不同，权重虽然一一对应，输出**不会**
与训练时一致，只适合作为继续微调的起点。

用法:
    python convert_to_hf.py                                   # last.pt -> ./hf_model
    python convert_to_hf.py --checkpoint ./checkpoints/last.pt --output ./hf_model
    python convert_to_hf.py --dtype float16                   # 半精度导出
    python convert_to_hf.py --verify                          # 与原始 checkpoint 比对 logits
    python convert_to_hf.py --format llama                    # 标准 Llama 结构（近似）

输出目录:
    config.json
    generation_config.json
    model.safetensors            （或 --weights-format bin 时为 pytorch_model.bin）
    configuration_universal_lm.py       （仅 custom）
    modeling_universal_lm.py            （仅 custom）
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    README.md
"""

import argparse
import json
import os
import shutil
import struct
import sys
from typing import Dict

import torch


PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hf_template')

DTYPE_MAP = {
    'float32': (torch.float32, 'float32'),
    'float16': (torch.float16, 'float16'),
    'bfloat16': (torch.bfloat16, 'bfloat16'),
}

SAFETENSORS_DTYPE = {
    torch.float32: 'F32',
    torch.float16: 'F16',
    torch.bfloat16: 'BF16',
    torch.float64: 'F64',
    torch.int64: 'I64',
    torch.int32: 'I32',
    torch.int16: 'I16',
    torch.int8: 'I8',
    torch.uint8: 'U8',
    torch.bool: 'BOOL',
}

ROPE_BUFFER_SUFFIXES = ('rope.inv_freq', 'rope.cos_cached', 'rope.sin_cached')


# ==================== safetensors 写出（不依赖 safetensors 库） ====================
def save_safetensors(path: str, tensors: Dict[str, torch.Tensor]) -> None:
    header: Dict[str, dict] = {'__metadata__': {'format': 'pt'}}
    offset = 0
    items = []
    for name, tensor in tensors.items():
        t = tensor.detach().contiguous().cpu()
        if t.dtype not in SAFETENSORS_DTYPE:
            raise TypeError(f'不支持的 dtype: {name} {t.dtype}')
        nbytes = t.numel() * t.element_size()
        header[name] = {
            'dtype': SAFETENSORS_DTYPE[t.dtype],
            'shape': list(t.shape),
            'data_offsets': [offset, offset + nbytes],
        }
        offset += nbytes
        items.append((name, t, nbytes))

    header_bytes = json.dumps(
        header, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    header_bytes += b' ' * ((-len(header_bytes)) % 8)

    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(header_bytes)))
        f.write(header_bytes)
        for name, t, nbytes in items:
            raw = t.flatten().view(torch.uint8).numpy().tobytes()
            if len(raw) != nbytes:
                raise RuntimeError(f'{name} 序列化长度不一致: {len(raw)} != {nbytes}')
            f.write(raw)


# ==================== 自定义架构（精确复现） ====================
def build_custom_state_dict(sd: Dict[str, torch.Tensor], cfg: dict,
                            dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in sd.items():
        if any(name.endswith(s) for s in ROPE_BUFFER_SUFFIXES):
            continue  # RoPE 由 modeling_universal_lm.py 按 config 重算
        if name == 'fc_out.weight' and bool(cfg.get('tie_weights', True)):
            continue
        out[f'model.{name}'] = tensor.to(dtype)

    if 'model.embedding.weight' not in out:
        raise RuntimeError('缺少 embedding.weight')
    if 'lm_head.weight' not in out:
        if bool(cfg.get('tie_weights', True)):
            out['lm_head.weight'] = out['model.embedding.weight']
        else:
            raise RuntimeError('非共享权重缺少 fc_out.weight')
    return out


def build_custom_config(cfg: dict, dtype_name: str) -> dict:
    if not bool(cfg.get('use_rope', True)):
        raise ValueError('当前转换器只支持 use_rope=True')
    return {
        'architectures': ['UniversalLMForCausalLM'],
        'model_type': 'universal_lm',
        'auto_map': {
            'AutoConfig': 'configuration_universal_lm.UniversalLMConfig',
            'AutoModelForCausalLM': 'modeling_universal_lm.UniversalLMForCausalLM',
        },
        'vocab_size': int(cfg['vocab_size']),
        'hidden_size': int(cfg['d_model']),
        'num_hidden_layers': int(cfg['n_layers']),
        'num_attention_heads': int(cfg['n_heads']),
        'intermediate_size': int(cfg['d_ff']),
        'max_position_embeddings': int(cfg['max_seq_len']),
        'hidden_act': 'silu',
        'rms_norm_eps': 1e-6,
        'rope_theta': 10000.0,
        'attention_dropout': 0.0,
        'initializer_range': 0.02,
        'tie_word_embeddings': bool(cfg.get('tie_weights', True)),
        'use_cache': True,
        # 同时写 torch_dtype（4.x）和 dtype（5.x），两个版本都能识别
        'torch_dtype': dtype_name,
        'dtype': dtype_name,
        'bos_token_id': BOS_ID,
        'eos_token_id': EOS_ID,
        'pad_token_id': PAD_ID,
        'unk_token_id': UNK_ID,
    }


# ==================== 标准 Llama 架构（近似） ====================
def build_llama_state_dict(sd: Dict[str, torch.Tensor], cfg: dict,
                           dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    d_model = int(cfg['d_model'])
    n_layers = int(cfg['n_layers'])
    out: Dict[str, torch.Tensor] = {}

    def put(name: str, tensor: torch.Tensor):
        if name in out:
            raise KeyError(f'重复的导出权重: {name}')
        out[name] = tensor.to(dtype)

    for name, tensor in sd.items():
        if any(name.endswith(s) for s in ROPE_BUFFER_SUFFIXES):
            continue

        if name == 'embedding.weight':
            put('model.embed_tokens.weight', tensor)
            if bool(cfg.get('tie_weights', True)):
                put('lm_head.weight', tensor)
            continue
        if name == 'norm.weight':
            put('model.norm.weight', tensor)
            continue
        if name == 'fc_out.weight':
            if bool(cfg.get('tie_weights', True)):
                continue
            put('lm_head.weight', tensor)
            continue

        if not name.startswith('layers.'):
            raise KeyError(f'未知权重: {name}')
        parts = name.split('.')
        layer = int(parts[1])
        module = parts[2]
        param = '.'.join(parts[3:])
        prefix = f'model.layers.{layer}'

        if module == 'self_attn' and param == 'wqkv.weight':
            if tensor.shape != (3 * d_model, d_model):
                raise ValueError(f'wqkv 形状异常: {tuple(tensor.shape)}')
            q, k, v = tensor.split(d_model, dim=0)
            put(f'{prefix}.self_attn.q_proj.weight', q)
            put(f'{prefix}.self_attn.k_proj.weight', k)
            put(f'{prefix}.self_attn.v_proj.weight', v)
        elif module == 'self_attn' and param == 'wo.weight':
            put(f'{prefix}.self_attn.o_proj.weight', tensor)
        elif module == 'ff' and param == 'w1.weight':
            put(f'{prefix}.mlp.gate_proj.weight', tensor)
        elif module == 'ff' and param == 'w3.weight':
            put(f'{prefix}.mlp.up_proj.weight', tensor)
        elif module == 'ff' and param == 'w2.weight':
            put(f'{prefix}.mlp.down_proj.weight', tensor)
        elif module == 'norm1' and param == 'weight':
            put(f'{prefix}.input_layernorm.weight', tensor)
        elif module == 'norm2' and param == 'weight':
            put(f'{prefix}.post_attention_layernorm.weight', tensor)
        else:
            raise KeyError(f'未知权重: {name}')

    missing = set(range(n_layers)) - {int(n.split('.')[2]) for n in out
                                      if n.startswith('model.layers.')}
    if missing:
        raise RuntimeError(f'缺少层: {sorted(missing)}')
    if 'lm_head.weight' not in out:
        raise RuntimeError('缺少 lm_head / fc_out 权重')
    return out


def build_llama_config(cfg: dict, dtype_name: str) -> dict:
    d_model = int(cfg['d_model'])
    n_heads = int(cfg['n_heads'])
    if d_model % n_heads != 0:
        raise ValueError('d_model 必须能被 n_heads 整除')
    if not bool(cfg.get('use_rope', True)):
        raise ValueError('当前转换器只支持 use_rope=True')
    return {
        'architectures': ['LlamaForCausalLM'],
        'model_type': 'llama',
        'vocab_size': int(cfg['vocab_size']),
        'hidden_size': d_model,
        'intermediate_size': int(cfg['d_ff']),
        'num_hidden_layers': int(cfg['n_layers']),
        'num_attention_heads': n_heads,
        'num_key_value_heads': n_heads,
        'head_dim': d_model // n_heads,
        'hidden_act': 'silu',
        'max_position_embeddings': int(cfg['max_seq_len']),
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'rope_theta': 10000.0,
        'rope_scaling': None,
        'attention_bias': False,
        'attention_dropout': 0.0,
        'mlp_bias': False,
        'pretraining_tp': 1,
        'tie_word_embeddings': bool(cfg.get('tie_weights', True)),
        'torch_dtype': dtype_name,
        'dtype': dtype_name,
        'bos_token_id': BOS_ID,
        'eos_token_id': EOS_ID,
        'pad_token_id': PAD_ID,
        'unk_token_id': UNK_ID,
    }


# ==================== 公共文件 ====================
def build_generation_config() -> dict:
    # 默认采样参数与 inf.py 对齐；HF 的 repetition_penalty 作用于全部历史。
    return {
        'bos_token_id': BOS_ID,
        'eos_token_id': EOS_ID,
        'pad_token_id': PAD_ID,
        'do_sample': True,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'repetition_penalty': 1.15,
        'max_new_tokens': 200,
    }


def write_tokenizer(src: str, out_dir: str, model_max_len: int) -> None:
    """复制训练用的 tokenizer.json，并补充 HF 所需的配置文件。"""
    with open(src, 'r', encoding='utf-8') as f:
        tok = json.load(f)
    vocab = tok.get('model', {}).get('vocab', {})
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            raise RuntimeError(f'tokenizer 词表缺少 {token}')

    shutil.copyfile(src, os.path.join(out_dir, 'tokenizer.json'))

    with open(os.path.join(out_dir, 'special_tokens_map.json'), 'w',
              encoding='utf-8') as f:
        json.dump({
            'bos_token': '<bos>',
            'eos_token': '<eos>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, 'tokenizer_config.json'), 'w',
              encoding='utf-8') as f:
        json.dump({
            'tokenizer_class': 'PreTrainedTokenizerFast',
            'bos_token': '<bos>',
            'eos_token': '<eos>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'add_bos_token': False,
            'add_eos_token': False,
            'clean_up_tokenization_spaces': False,
            'model_max_length': model_max_len,
            'added_tokens_decoder': {
                str(PAD_ID): {'content': '<pad>', 'special': True},
                str(UNK_ID): {'content': '<unk>', 'special': True},
                str(BOS_ID): {'content': '<bos>', 'special': True},
                str(EOS_ID): {'content': '<eos>', 'special': True},
            },
        }, f, ensure_ascii=False, indent=2)


def write_readme(out_dir: str, cfg: dict, dtype_name: str, weights_file: str,
                 fmt: str) -> None:
    abs_dir = os.path.abspath(out_dir)
    if fmt == 'custom':
        load_line = ('model = AutoModelForCausalLM.from_pretrained(\n'
                     '    path, trust_remote_code=True, dtype="auto"\n'
                     ').to("cuda" if torch.cuda.is_available() else "cpu")')
        arch_note = (
            '使用自定义架构 `UniversalLMForCausalLM`，逐位复现训练代码（包括本项目特殊的\n'
            'RoPE 实现），因此加载时必须传 `trust_remote_code=True`。')
    else:
        load_line = ('model = AutoModelForCausalLM.from_pretrained(\n'
                     '    path, dtype="auto"\n'
                     ').to("cuda" if torch.cuda.is_available() else "cpu")')
        arch_note = (
            '⚠️ 使用标准 `LlamaForCausalLM` 结构。**权重不会逐位复现训练输出**：\n'
            '本项目 RoPE 在完整 hidden_size 上计算 inv_freq 后截断到 head_dim，\n'
            '与标准 Llama 的 per-head RoPE 不同。仅建议作为继续微调的起点。')

    text = f"""# Universal LM (Hugging Face 格式)

由 `convert_to_hf.py` 从通用训练脚本的 checkpoint 转换而来。

{arch_note}

## 结构

| 项 | 值 |
|---|---|
| hidden_size | {cfg['d_model']} |
| layers | {cfg['n_layers']} |
| heads | {cfg['n_heads']} |
| intermediate_size | {cfg['d_ff']} |
| vocab | {cfg['vocab_size']} |
| max_seq_len | {cfg['max_seq_len']} |
| dtype | {dtype_name} |
| 权重文件 | `{weights_file}` |

## 使用

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "{abs_dir}"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
{load_line}
model.eval()

prompt = "人工智能是"
ids = [tok.bos_token_id] + tok.encode(prompt)   # 训练时文档以 <bos> 开头
input_ids = torch.tensor([ids], device=model.device)

with torch.no_grad():
    out = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.15,
        eos_token_id={EOS_ID},
        pad_token_id={PAD_ID},
    )
print(tok.decode(out[0][len(ids):], skip_special_tokens=True))
```

> HF 的 `repetition_penalty` 作用于全部上下文；本项目 `inf.py` 默认只回看
> 最近 512 个 token，因此两者结果不会逐 token 一致。
>
> transformers 4.x 请把 `dtype="auto"` 换成 `torch_dtype="auto"`。

## 特殊 token

| token | id |
|---|---|
| `<pad>` | {PAD_ID} |
| `<unk>` | {UNK_ID} |
| `<bos>` | {BOS_ID} |
| `<eos>` | {EOS_ID} |

## 注意

- 训练时每篇文档为 `[BOS] + 正文 + [EOS]`，续写时建议手动前置 `bos_token_id`。
- 训练代码刻意让文本里字面出现的 `<bos>` / `<eos>` 走普通 BPE；而 HF
  tokenizer 会把它们识别为特殊 token（这是 HF 的行为，无法只对 decode 生效）。
  如果提示词里包含这两个字面串，请在送入模型前自行转义/替换。
"""
    with open(os.path.join(out_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(text)


# ==================== 校验 ====================
def verify(cfg: dict, hf_dir: str, checkpoint: str, fmt: str,
           dtype_name: str = 'float32', n_tokens: int = 32) -> None:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print('⚠️ 未安装 transformers，跳过 --verify')
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import inf  # 复用推理脚本里的参考模型定义

    torch.manual_seed(0)
    ids = torch.randint(4, int(cfg['vocab_size']), (1, n_tokens), dtype=torch.long)

    ref = inf.DecoderOnlyTransformer(inf.ModelConfig(**cfg)).eval()
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    ref.load_state_dict(ckpt['model_state_dict'])

    kwargs = {}
    if fmt == 'custom':
        kwargs['trust_remote_code'] = True
    hf = AutoModelForCausalLM.from_pretrained(
        hf_dir, dtype='auto', **kwargs).eval()
    if hf.dtype != torch.float32:
        hf = hf.float()

    with torch.no_grad():
        ref_logits = ref(ids)[0, -1]
        hf_logits = hf(ids).logits[0, -1]

    diff = (ref_logits - hf_logits).abs()
    max_diff = float(diff.max())
    cos = torch.nn.functional.cosine_similarity(
        ref_logits.unsqueeze(0), hf_logits.unsqueeze(0)).item()

    print(f'\n[verify] logits max|Δ|={max_diff:.3e} '
          f'mean|Δ|={float(diff.mean()):.3e} cos={cos:.8f}')
    if fmt == 'custom':
        threshold = 2e-3 if dtype_name == 'float32' else 1.5e-1
    else:
        threshold = 1e-1
    if max_diff > threshold:
        print('⚠️ 差异偏大，请检查转换映射')
    elif fmt == 'custom' and dtype_name != 'float32':
        print(f'✓ 半精度导出（{dtype_name}），差异来自量化')
    else:
        print('✓ HF 模型与原始 checkpoint 输出一致'
              if fmt == 'custom' else '✓ 权重映射成功（RoPE 差异见 README）')


# ==================== 主流程 ====================
def parse_args():
    p = argparse.ArgumentParser(description='把训练 checkpoint 转换为本地 HF 模型')
    p.add_argument('--checkpoint', type=str, default='./checkpoints/last.pt')
    p.add_argument('--tokenizer', type=str, default='./tokenizer.json')
    p.add_argument('--output', type=str, default='./hf_model')
    p.add_argument('--format', choices=['custom', 'llama'], default='custom',
                   help='custom=精确复现（默认）；llama=标准结构（近似）')
    p.add_argument('--dtype', choices=list(DTYPE_MAP), default='float32')
    p.add_argument('--weights-format', choices=['safetensors', 'bin'],
                   default='safetensors')
    p.add_argument('--overwrite', action='store_true', help='输出目录已存在时清空重建')
    p.add_argument('--verify', action='store_true', help='转换后加载 HF 模型并比对 logits')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f'❌ checkpoint 不存在: {args.checkpoint}')
    if not os.path.exists(args.tokenizer):
        raise SystemExit(f'❌ tokenizer 不存在: {args.tokenizer}')
    if args.format == 'custom' and not os.path.isdir(TEMPLATE_DIR):
        raise SystemExit(f'❌ 缺少模板目录: {TEMPLATE_DIR}')

    out_dir = args.output
    if os.path.isdir(out_dir):
        if os.listdir(out_dir):
            if not args.overwrite:
                raise SystemExit(f'❌ 输出目录非空: {out_dir}（加 --overwrite 覆盖）')
            shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'加载 checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in ckpt or 'config' not in ckpt:
        raise SystemExit('❌ checkpoint 缺少 model_state_dict / config')

    cfg = dict(ckpt['config'])
    print(f'模型: d_model={cfg["d_model"]} layers={cfg["n_layers"]} '
          f'heads={cfg["n_heads"]} ff={cfg["d_ff"]} vocab={cfg["vocab_size"]}')

    dtype, dtype_name = DTYPE_MAP[args.dtype]
    if args.format == 'custom':
        state = build_custom_state_dict(ckpt['model_state_dict'], cfg, dtype)
        hf_cfg = build_custom_config(cfg, dtype_name)
        for fn in ('configuration_universal_lm.py', 'modeling_universal_lm.py'):
            shutil.copyfile(os.path.join(TEMPLATE_DIR, fn),
                            os.path.join(out_dir, fn))
        n_unique = sum(v.numel() for k, v in state.items() if k != 'lm_head.weight')
        n_total = sum(v.numel() for v in state.values())
        print(f'格式: custom (UniversalLMForCausalLM)  参数 {n_unique/1e6:.1f}M '
              f'(含共享 lm_head 共 {n_total/1e6:.1f}M)')
    else:
        state = build_llama_state_dict(ckpt['model_state_dict'], cfg, dtype)
        hf_cfg = build_llama_config(cfg, dtype_name)
        n_total = sum(v.numel() for v in state.values())
        print(f'格式: llama (LlamaForCausalLM)  参数 {n_total/1e6:.1f}M')
        print('⚠️ 标准 Llama 的 RoPE 与训练代码不同，输出不会逐位一致')

    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(hf_cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'generation_config.json'), 'w',
              encoding='utf-8') as f:
        json.dump(build_generation_config(), f, ensure_ascii=False, indent=2)

    weights_file = ('model.safetensors' if args.weights_format == 'safetensors'
                    else 'pytorch_model.bin')
    weights_path = os.path.join(out_dir, weights_file)
    print(f'写出权重: {weights_file}')
    if args.weights_format == 'safetensors':
        save_safetensors(weights_path, state)
    else:
        torch.save(state, weights_path)

    write_tokenizer(args.tokenizer, out_dir, int(cfg['max_seq_len']))
    write_readme(out_dir, cfg, dtype_name, weights_file, args.format)

    print(f'✓ 已导出到: {os.path.abspath(out_dir)}')
    print(f'  权重大小: {os.path.getsize(weights_path)/1e6:.1f} MB')

    if args.verify:
        verify(cfg, out_dir, args.checkpoint, args.format, dtype_name)


if __name__ == '__main__':
    main()
