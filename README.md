# Universal LM Trainer

一个配置驱动的 **Decoder-only 语言模型预训练 / 推理 / Hugging Face 导出** 项目。
同一份代码既可以训练几十 M 的小模型做实验，也可以训练上百 M 的模型做正式预训练，
不需要改 Python 源码，只改 `config.json`。

本仓库最初用于在 **NVIDIA V100 (sm70)** 上从零训练一个中文 / 英文 / 代码模型，
但训练逻辑与模型规模解耦，仓库里带了几份不同大小的预设配置。

## 特性

- **完全配置驱动**：模型宽度、深度、FFN、上下文长度、词表、batch、学习率、
  路径全部写在 JSON 配置里，CLI 只做少量覆盖。
- **真实 FlashAttention-2**：`csrc/flash_attn_v100/` 提供 sm70 CUDA 前向 / 反向
  kernel，不依赖 xformers；不可用时自动回退 PyTorch SDPA。
- **平铺 uint16 memmap 缓存**：把分桶样本转成 `bucket_*.tokens.u16` +
  `offsets.npy`，单样本随机读约 2KB，避免每个样本反序列化整个 chunk。
- **长度分桶 + Token-Budget 微批**：桶边界由 `max_seq_len` 自动生成，短序列不会
  让 GPU 空转；每个优化器 step 严格消费固定样本数。
- **分块 CrossEntropy**：不物化 `[B*S, vocab]` 的 fp32 logits。
- **fused AdamW / 融合 RMSNorm / torch.compile**：V100 上尽量榨性能。
- **BOS/EOS 监督**：每个样本编码为 `[BOS] + 正文 + [EOS]`，可用
  `--no-bos` / `--no-eos` 关闭；文本里字面出现的 `<bos>` / `<eos>` 按普通
  BPE 处理，不会被当成特殊 token。
- **可恢复训练**：Ctrl+C 菜单保存完整状态（模型 / 优化器 / scaler / 调度器 /
  RNG / 已消费样本位置），`--resume` 严格不重不漏；数据指纹不匹配时拒绝恢复。
- **推理与评分**：流式生成、重复惩罚、禁止重复 n-gram、分块困惑度评分。
- **HF 导出**：一条命令导出本地 Hugging Face 模型，默认导出可逐位复现训练实现的
  自定义架构（`trust_remote_code=True`）。

## 目录结构

```
.
├── train.py                        # 训练主脚本（分词器 / 缓存 / 训练 / 续训）
├── inf.py                          # 推理与评分（单文件，不依赖 train.py）
├── convert_to_hf.py                # checkpoint -> Hugging Face 模型目录
├── config.json                     # 默认配置（100M 预设）
├── configs/
│   ├── tiny-test.json              # 极小模型，CPU 也能跑通流程
│   ├── 50m.json
│   ├── 100m.json
│   └── 300m.json
├── hf_template/
│   ├── configuration_universal_lm.py
│   └── modeling_universal_lm.py
├── flash_attn_v100.py              # FlashAttention-2 Python 封装
├── flash_attn_v100_cuda.py         # 已编译扩展加载器
├── csrc/flash_attn_v100/           # CUDA kernels（sm70）
├── repair_cache.py                 # 缓存计数 / 边界检查
├── repair_cache_special_tokens.py  # 清理旧缓存里误编码的特殊 token
├── tok_test.py                     # 分词器测试
├── requirements.txt
└── .gitignore
```

> 训练数据、`tokenizer.json`、`checkpoints/`、`cache/`、`hf_model/` 体积较大，
> 未包含在仓库中，需要自行生成。

## 环境

```bash
pip install torch tokenizers numpy tqdm psutil
# HF 导出 / 加载（可选）
pip install transformers safetensors
# Triton 版本 FlashAttention（可选，非 V100 CUDA 扩展路径）
pip install triton
```

V100 的 CUDA 扩展需要匹配的 CUDA 工具链。首次运行时会自动编译
`csrc/flash_attn_v100/build/flash_attn_v100_cuda.so`。没有 V100 / 编译失败时
会自动回退 SDPA，也可以显式禁用：

```bash
FA_V100_DISABLE=1 python train.py
```

## 快速开始

### 1. 准备数据

把语料放到 `data/*.jsonl`，每行一个 JSON，字段为 `text` 或 `content`：

```json
{"text": "一篇完整文档的正文……"}
```

每个样本会被当成一篇完整文档，按 `max_seq_len` 截断正文。

### 2. 选配置

```bash
# 默认配置：100M（d_model=768, 10 层, 12 头, 4096 上下文）
python train.py

# 其他预设
python train.py --config configs/50m.json
python train.py --config configs/300m.json

# 先用极小模型在 CPU 上跑通流程
python train.py --config configs/tiny-test.json --max-steps 1
```

### 3. 训练

```bash
# 首次运行：自动训练 tokenizer、构建缓存、开始训练
python train.py --config config.json

# 强制重训 tokenizer（会改变 token id，不能与旧 checkpoint 混用）
python train.py --retrain-tokenizer

# 只构建数据缓存
python train.py --cache-only

# 从 <checkpoint_dir>/last.pt 恢复
python train.py --resume

# 只跑指定 step 数（调试 / 基准）
python train.py --max-steps 100
```

训练过程中按 `Ctrl+C` 会弹出保存菜单。只有主动保存和 epoch 结束才会写
`<checkpoint_dir>/last.pt`。

### 4. 推理

```bash
python inf.py --model ./checkpoints/last.pt --tokenizer ./tokenizer.json \
    --prompt "人工智能是" --max_new_tokens 200 --device cuda --fp16

python inf.py --model ./checkpoints/last.pt --interactive
python inf.py --model ./checkpoints/last.pt --batch_file prompts.txt
```

重复惩罚默认开启（`--repetition_penalty 1.15 --repetition_window 512`，
`--no_repeat_ngram_size 0`）。关闭：`--repetition_penalty 1.0`。

### 5. 评分

```bash
python inf.py --model ./checkpoints/last.pt \
    --test data/test_samples/mixed.jsonl --test_detail --device cuda --fp16
```

`--test` 支持逗号分隔多个文件，并可按文件分类汇总。

### 6. 导出 Hugging Face 模型

```bash
python convert_to_hf.py --checkpoint ./checkpoints/last.pt \
    --tokenizer ./tokenizer.json --output ./hf_model --verify
```

导出目录可直接加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "./hf_model"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, dtype="auto"
)
```

> 本项目 RoPE 与标准 Llama 不同（先在完整 `hidden_size` 上计算 `inv_freq`，
> 再截断到 `head_dim`），因此默认导出的是自定义 `UniversalLMForCausalLM`，
> 不能直接当 `LlamaForCausalLM` 使用。`--format llama` 只适合作为继续微调的
> 近似起点，不保证逐位一致。

## 配置说明

`config.json` 分成四块：

### `model`

| 字段 | 说明 |
|---|---|
| `d_model` | 隐藏维度，必须能被 `n_heads` 整除 |
| `n_heads` | 注意力头数，`head_dim = d_model // n_heads` |
| `n_layers` | Transformer block 层数 |
| `d_ff` | FFN 中间维度（SwiGLU） |
| `max_seq_len` | 最大上下文长度，同时决定分桶边界与缓存 |
| `dropout` | dropout 概率 |
| `use_rope` | 是否使用 RoPE（默认 true） |
| `tie_weights` | 输入 embedding 与 `lm_head` 是否共享权重 |
| `use_gradient_checkpointing` | 是否开启梯度检查点（省显存、慢一点） |

### `training`

| 字段 | 说明 |
|---|---|
| `batch_size` | 每个优化器 step 的样本数 |
| `gradient_accumulation_steps` | 梯度累积步数；实际每步样本数 = `batch_size * grad_accum` |
| `epochs` | 训练轮数 |
| `learning_rate` / `weight_decay` / `max_grad_norm` | 优化器参数 |
| `warmup_steps` / `min_lr_ratio` | 学习率调度 |
| `save_every` / `log_every` | 保存 / 日志频率 |
| `num_workers` / `prefetch_factor` | DataLoader 并行度；CPU 调试时设 `num_workers: 0` |
| `seed` | 随机种子，同时参与缓存 ID 与数据指纹 |
| `use_fp16` | GPU 上是否使用 FP16 混合精度 |
| `use_compile` | 是否 `torch.compile`，也可用 `UNIVERSAL_LM_COMPILE=0` 关闭 |
| `fused_optimizer` | 是否使用 fused AdamW；CPU 或旧版 PyTorch 请设 false |
| `ce_chunk_rows` | 分块 cross-entropy 的行数 |
| `target_tokens` | 每个微批的 token 预算 |
| `max_micro_batch` | 微批样本数上限（也等于每步最大样本数） |
| `test_samples` | `--test-mode` 时最多使用的样本数 |
| `prepend_bos` / `append_eos` | 是否加 BOS / EOS 监督 |
| `max_steps` | >0 时只跑指定 step 数 |

### `tokenizer`

| 字段 | 说明 |
|---|---|
| `vocab_size` | BPE 目标词表大小（必须 ≤ 65536，缓存是 uint16） |
| `path` | `tokenizer.json` 路径 |
| `max_bytes` | 训练 tokenizer 时采样的语料字节预算 |
| `min_frequency` | BPE 最小词频 |

### `paths`

| 字段 | 说明 |
|---|---|
| `data_dir` | 存放 `*.jsonl` 的数据目录 |
| `cache_dir` | 分桶缓存根目录；实际缓存在 `<cache_dir>/<cache_id>/flat_u16` |
| `checkpoint_dir` | checkpoint 目录 |

## CLI 覆盖

常用参数可以临时覆盖配置，不必改 JSON：

```bash
python train.py \
    --config config.json \
    --data-dir ./data \
    --cache-dir ./cache \
    --checkpoint-dir ./checkpoints \
    --tokenizer ./tokenizer.json \
    --resume \
    --max-steps 1000 \
    --seed 1234
```

其他开关：`--retrain-tokenizer`、`--cache-only`、`--no-bos`、`--no-eos`、
`--test-mode`。

## 缓存与换模型

缓存 ID 只由 `tokenizer 大小 + max_seq_len + 分桶边界 + seed` 决定，**与模型结构
无关**。因此：

- 用同一份 tokenizer / `max_seq_len` / `seed` 训练不同大小的模型，可以直接复用
  同一份缓存；
- 换 tokenizer、改 `max_seq_len` 或改 `seed` 会生成新的 `cache_id`，旧缓存不会被
  误用；
- `--resume` 会校验数据指纹（tokenizer + 分桶计数 + BOS/EOS 开关），不匹配就
  拒绝恢复，避免 token id 错位。

例如要把之前 `data/bucketed_cache_streaming_4096` 下的缓存接到本仓库：

```bash
python train.py --config config.json \
    --cache-dir data/bucketed_cache_streaming_4096
```

只要 `tokenizer.json`、`max_seq_len=4096`、`seed=42`、BOS/EOS 设置与旧训练一致，
`cache_id` 就会完全一致，直接复用 20GB 平铺缓存。

## 性能与环境变量

| 变量 | 作用 |
|---|---|
| `UNIVERSAL_LM_COMPILE=0` | 关闭 `torch.compile`（兼容 `EMINI_COMPILE=0`） |
| `FA_V100_DISABLE=1` | 禁用 V100 CUDA FlashAttention 扩展，回退 SDPA |
| `CACHE_BATCH_LINES` | JSONL 建缓存时每批读取行数 |

建议：

- V100 上保持 `use_fp16: true`、`use_compile: true`、`fused_optimizer: true`；
- 显存不足时先降 `target_tokens` / `max_micro_batch`，再考虑
  `use_gradient_checkpointing: true`；
- CPU 调试用 `configs/tiny-test.json`，并设置
  `FA_V100_DISABLE=1 UNIVERSAL_LM_COMPILE=0`。

## 常见问题

**Q: 词表能超过 65536 吗？**
当前缓存格式是 uint16，不能。需要改缓存格式（例如 uint32）后才能支持更大词表。

**Q: 为什么导出的是自定义模型而不是 Llama？**
见上文：RoPE 的 `inv_freq` 计算方式与标准 Llama 不同。要得到逐位一致的 logits，
必须用 `hf_template/` 里的自定义实现。

**Q: 训练中断后怎么恢复？**
在 Ctrl+C 菜单里选保存，写入 `<checkpoint_dir>/last.pt`，然后
`python train.py --resume`。只有 step 边界会落盘，恢复后不重不漏。

**Q: 可以只用推理，不装 CUDA 扩展吗？**
可以。`inf.py` 是独立单文件，会自动回退 SDPA。

## License

见各源文件头部说明。使用前请确认你拥有相应授权。
