# Llama4 Usage

[Llama 4](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md) is Meta's latest generation of open-source LLM model with industry-leading performance.

**中文对照**：[Llama 4](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md) 是 Meta 最新一代的开源 LLM 模型，具有行业领先的性能。

SGLang has supported Llama 4 Scout (109B) and Llama 4 Maverick (400B) since [v0.4.5](https://github.com/sgl-project/sglang/releases/tag/v0.4.5).

**中文对照**：SGLang 自 [v0.4.5](https://github.com/sgl-project/sglang/releases/tag/v0.4.5) 起支持 Llama 4 Scout (109B) 和 Llama 4 Maverick (400B)。

Ongoing optimizations are tracked in the [Roadmap](https://github.com/sgl-project/sglang/issues/5118).

**中文对照**：正在进行的优化在[路线图](https://github.com/sgl-project/sglang/issues/5118)中跟踪。

## Launch Llama 4 with SGLang

**中文对照**：使用 SGLang 启动 Llama 4

To serve Llama 4 models on 8xH100/H200 GPUs:

**中文对照**：要在 8xH100/H200 GPU 上服务 Llama 4 模型：

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tp 8 \
  --context-length 1000000
```

### Configuration Tips

**中文对照**：配置提示

- **OOM Mitigation**: Adjust `--context-length` to avoid a GPU out-of-memory issue. For the Scout model, we recommend setting this value up to 1M on 8*H100 and up to 2.5M on 8*H200. For the Maverick model, we don't need to set context length on 8*H200. When hybrid kv cache is enabled, `--context-length` can be set up to 5M on 8*H100 and up to 10M on 8*H200 for the Scout model.

**中文对照**：**OOM 缓解**：调整 `--context-length` 以避免 GPU 内存不足问题。对于 Scout 模型，我们建议在 8*H100 上将此值设置为最高 1M，在 8*H200 上设置为最高 2.5M。对于 Maverick 模型，在 8*H200 上不需要设置上下文长度。启用混合 kv 缓存时，Scout 模型的 `--context-length` 在 8*H100 上可设置为最高 5M，在 8*H200 上可设置为最高 10M。

- **Attention Backend Auto-Selection**: SGLang automatically selects the optimal attention backend for Llama 4 based on your hardware. You typically don't need to specify `--attention-backend` manually:

**中文对照**：**注意力后端自动选择**：SGLang 根据你的硬件自动为 Llama 4 选择最优的注意力后端。通常你不需要手动指定 `--attention-backend`：

  - **Blackwell GPUs (B200/GB200)**: `trtllm_mha`

**中文对照**：- **Blackwell GPU (B200/GB200)**：`trtllm_mha`

  - **Hopper GPUs (H100/H200)**: `fa3`

**中文对照**：- **Hopper GPU (H100/H200)**：`fa3`

  - **AMD GPUs**: `aiter`

**中文对照**：- **AMD GPU**：`aiter`

  - **Intel XPU**: `intel_xpu`

**中文对照**：- **Intel XPU**：`intel_xpu`

  - **Other platforms**: `triton` (fallback)

**中文对照**：- **其他平台**：`triton`（回退）

  To override the auto-selection, explicitly specify `--attention-backend` with one of the supported backends: `fa3`, `aiter`, `triton`, `trtllm_mha`, or `intel_xpu`.

**中文对照**：要覆盖自动选择，请使用以下支持的后端之一显式指定 `--attention-backend`：`fa3`、`aiter`、`triton`、`trtllm_mha` 或 `intel_xpu`。

- **Chat Template**: Add `--chat-template llama-4` for chat completion tasks.

**中文对照**：**聊天模板**：添加 `--chat-template llama-4` 以用于聊天完成任务。

- **Enable Multi-Modal**: Add `--enable-multimodal` for multi-modal capabilities.

**中文对照**：**启用多模态**：添加 `--enable-multimodal` 以获得多模态能力。

- **Enable Hybrid-KVCache**: Set `--swa-full-tokens-ratio` to adjust the ratio of SWA layer (for Llama4, it's local attention layer) KV tokens / full layer KV tokens. (default: 0.8, range: 0-1)

**中文对照**：**启用混合 KVCache**：设置 `--swa-full-tokens-ratio` 以调整 SWA 层（对于 Llama4，它是局部注意力层）KV 令牌 / 全层 KV 令牌的比率。（默认值：0.8，范围：0-1）


### EAGLE Speculative Decoding

**中文对照**：EAGLE 推测解码

**Description**: SGLang has supported Llama 4 Maverick (400B) with [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding).

**中文对照**：**描述**：SGLang 通过 [EAGLE 推测解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding) 支持 Llama 4 Maverick (400B)。

**Usage**:

**中文对照**：**用法**：

Add arguments `--speculative-draft-model-path`, `--speculative-algorithm`, `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` to enable this feature. For example:

**中文对照**：添加参数 `--speculative-draft-model-path`、`--speculative-algorithm`、`--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 以启用此功能。例如：
```
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path nvidia/Llama-4-Maverick-17B-128E-Eagle3 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --trust-remote-code \
  --tp 8 \
  --context-length 1000000
```

- **Note** The Llama 4 draft model *nvidia/Llama-4-Maverick-17B-128E-Eagle3* can only recognize conversations in chat mode.

**中文对照**：- **注意** Llama 4 草稿模型 *nvidia/Llama-4-Maverick-17B-128E-Eagle3* 只能识别聊天模式下的对话。

## Benchmarking Results

**中文对照**：基准测试结果

### Accuracy Test with `lm_eval`

**中文对照**：使用 `lm_eval` 进行准确率测试

The accuracy on SGLang for both Llama4 Scout and Llama4 Maverick can match the [official benchmark numbers](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).

**中文对照**：SGLang 上 Llama4 Scout 和 Llama4 Maverick 的准确率可以匹配[官方基准数字](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)。

Benchmark results on MMLU Pro dataset with 8*H100:

**中文对照**：在 8*H100 上 MMLU Pro 数据集的基准测试结果：

|                    | Llama-4-Scout-17B-16E-Instruct | Llama-4-Maverick-17B-128E-Instruct  |
|--------------------|--------------------------------|-------------------------------------|
| Official Benchmark | 74.3                           | 80.5                                |
| SGLang             | 75.2                           | 80.7                                |

Commands:

**中文对照**：命令：

```bash
# Llama-4-Scout-17B-16E-Instruct model
python -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --port 30000 \
  --tp 8 \
  --mem-fraction-static 0.8 \
  --context-length 65536
lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Scout-17B-16E-Instruct,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0

# Llama-4-Maverick-17B-128E-Instruct
python -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --port 30000 \
  --tp 8 \
  --mem-fraction-static 0.8 \
  --context-length 65536
lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Maverick-17B-128E-Instruct,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0
```

Details can be seen in [this PR](https://github.com/sgl-project/sglang/pull/5092).

**中文对照**：详情可见于[此 PR](https://github.com/sgl-project/sglang/pull/5092)。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/llama4.py` | Llama 4 模型架构：MoE 层 + 混合 SWA/全注意力 |
| `python/sglang/srt/layers/attention/` | 注意力后端自动选择：FA3（Hopper）、trtllm_mha（Blackwell）、aiter（AMD）、triton（回退） |
| `python/sglang/srt/speculative/eagle_worker.py` | Llama 4 Maverick 的 EAGLE3 推测解码 |
| `python/sglang/srt/server_args.py` | 定义 `--context-length`、`--swa-full-tokens-ratio`、`--chat-template`、`--enable-multimodal` 参数 |

### 关键代码逻辑

- **混合 KV 缓存**：`--swa-full-tokens-ratio` 控制 SWA（局部注意力）与全 KV 令牌的比率；减少长上下文服务的内存消耗
- **注意力自动选择**：`model_config.py` 检测 GPU 架构并选择最优注意力后端（FA3/trtllm_mha/aiter/triton）
- **EAGLE3 推测解码**：使用外部草稿模型（`nvidia/Llama-4-Maverick-17B-128E-Eagle3`）实现加速解码

### 集成要点

- **服务器参数**：`--tp 8`、`--context-length`（混合 KV 下最高 10M）、`--chat-template llama-4`、`--enable-multimodal`、`--swa-full-tokens-ratio`
- **推测解码**：`--speculative-algorithm EAGLE3 --speculative-draft-model-path nvidia/Llama-4-Maverick-17B-128E-Eagle3`
