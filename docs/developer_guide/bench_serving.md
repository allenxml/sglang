# Bench Serving Guide

This guide explains how to benchmark online serving throughput and latency using `python -m sglang.bench_serving`. It supports multiple inference backends via OpenAI-compatible and native endpoints, and produces both console metrics and optional JSONL outputs.

**中文对照**：本指南介绍如何使用 `python -m sglang.bench_serving` 对在线服务的吞吐量和延迟进行基准测试。它通过 OpenAI 兼容和原生端点支持多种推理后端，并生成控制台指标和可选的 JSONL 输出。

### What it does

- Generates synthetic or dataset-driven prompts and submits them to a target serving endpoint
- Measures throughput, time-to-first-token (TTFT), inter-token latency (ITL), per-request end-to-end latency, and more
- Supports streaming or non-streaming modes, rate control, and concurrency limits

**中文对照**：
- 生成合成或数据集驱动的提示词并将其提交到目标服务端点
- 测量吞吐量、首 token 时间（TTFT）、token 间延迟（ITL）、每请求端到端延迟等
- 支持流式或非流式模式、速率控制和并发限制

### Supported backends and endpoints

- `sglang` / `sglang-native`: `POST /generate`
- `sglang-oai`, `vllm`, `lmdeploy`: `POST /v1/completions`
- `sglang-oai-chat`, `vllm-chat`, `lmdeploy-chat`: `POST /v1/chat/completions`
- `trt` (TensorRT-LLM): `POST /v2/models/ensemble/generate_stream`
- `gserver`: Custom server (Not Implemented yet in this script)
- `truss`: `POST /v1/models/model:predict`

**中文对照**：
- `sglang` / `sglang-native`: `POST /generate`
- `sglang-oai`、`vllm`、`lmdeploy`: `POST /v1/completions`
- `sglang-oai-chat`、`vllm-chat`、`lmdeploy-chat`: `POST /v1/chat/completions`
- `trt` (TensorRT-LLM): `POST /v2/models/ensemble/generate_stream`
- `gserver`: 自定义服务器（此脚本中尚未实现）
- `truss`: `POST /v1/models/model:predict`

If `--base-url` is provided, requests are sent to it. Otherwise, `--host` and `--port` are used. When `--model` is not provided, the script will attempt to query `GET /v1/models` for an available model ID (OpenAI-compatible endpoints).

**中文对照**：如果提供了 `--base-url`，请求将发送到该地址。否则，使用 `--host` 和 `--port`。当未提供 `--model` 时，脚本将尝试查询 `GET /v1/models` 以获取可用的模型 ID（适用于 OpenAI 兼容端点）。

### Prerequisites

- Python 3.8+
- Dependencies typically used by this script: `aiohttp`, `numpy`, `requests`, `tqdm`, `transformers`, and for some datasets `datasets`, `pillow`, `pybase64`. Install as needed.
- An inference server running and reachable via the endpoints above
- If your server requires authentication, set environment variable `OPENAI_API_KEY` (used as `Authorization: Bearer <key>`)

**中文对照**：
- Python 3.8+
- 此脚本通常使用的依赖项：`aiohttp`、`numpy`、`requests`、`tqdm`、`transformers`，对于某些数据集还需要 `datasets`、`pillow`、`pybase64`。根据需要安装。
- 一个推理服务器正在运行，可以通过上述端点访问
- 如果服务器需要身份验证，请设置环境变量 `OPENAI_API_KEY`（用作 `Authorization: Bearer <key>`）

### Quick start

Run a basic benchmark against an sglang server exposing `/generate`:

**中文对照**：针对暴露 `/generate` 端点的 sglang 服务器运行基本基准测试：

```bash
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
```

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

Or, using an OpenAI-compatible endpoint (completions):

**中文对照**：或者，使用 OpenAI 兼容端点（补全）：

```bash
python3 -m sglang.bench_serving \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

### Datasets

Select with `--dataset-name`:

**中文对照**：
### 数据集

使用 `--dataset-name` 选择：

- `sharegpt` (default): loads ShareGPT-style pairs; optionally restrict with `--sharegpt-context-len` and override outputs with `--sharegpt-output-len`
- `random`: random text lengths; sampled from ShareGPT token space
- `random-ids`: 随机 token ID（可能产生乱码）
- `image`：生成图像并将其包装在聊天消息中；支持自定义分辨率、多种格式和不同内容类型
- `generated-shared-prefix`：具有共享长系统提示和短问题的合成数据集
- `mmmu`：从 MMMU（数学分割）采样并包含图像

**中文对照**：
- `random-ids`：随机 token ID（可能产生乱码）
- `image`：生成图像并将其包装在聊天消息中；支持自定义分辨率、多种格式和不同内容类型
- `generated-shared-prefix`：具有共享长系统提示和短问题的合成数据集
- `mmmu`：从 MMMU（数学分割）采样并包含图像

Common dataset flags:

- `--num-prompts N`: number of requests
- `--random-input-len`、`--random-output-len`、`--random-range-ratio`：用于 random/random-ids/image
- `--image-count`：每个请求的图像数量（用于 `image` 数据集）。

- `--apply-chat-template`：构建提示词时应用分词器聊天模板
- `--dataset-path PATH`：ShareGPT json 文件路径；如果为空且缺失，将被下载并缓存

**中文对照**：
- `--random-input-len`、`--random-output-len`、`--random-range-ratio`：用于 random/random-ids/image
- `--image-count`：每个请求的图像数量（用于 `image` 数据集）。
- `--apply-chat-template`：构建提示词时应用分词器聊天模板
- `--dataset-path PATH`：ShareGPT json 文件路径；如果为空且缺失，将被下载并缓存

Generated Shared Prefix flags (for `generated-shared-prefix`):

- `--gsp-num-groups`
- `--gsp-prompts-per-group`
- `--gsp-system-prompt-len`
- `--gsp-question-len`
- `--gsp-output-len`

**中文对照**：
- `--gsp-prompts-per-group`
- `--gsp-system-prompt-len`
- `--gsp-question-len`
- `--gsp-output-len`

Image dataset flags (for `image`):

- `--image-count`：每个请求的图像数量
- `--image-resolution`：图像分辨率；支持预设（4k、1080p、720p、360p）或自定义 '高度x宽度' 格式（例如 1080x1920、512x768）
- `--image-format`：图像格式（jpeg 或 png）
- `--image-content`：图像内容类型（random 或 blank）

**中文对照**：
- `--image-count`：每个请求的图像数量
- `--image-resolution`：图像分辨率；支持预设（4k、1080p、720p、360p）或自定义 '高度x宽度' 格式（例如 1080x1920、512x768）
- `--image-format`：图像格式（jpeg 或 png）
- `--image-content`：图像内容类型（random 或 blank）

### Examples

1. To benchmark image dataset with 3 images per request, 500 prompts, 512 input length, and 512 output length, you can run:

**中文对照**：要对图像数据集进行基准测试，每个请求 3 张图像，500 个提示词，512 输入长度和 512 输出长度，可以运行：

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-3B-Instruct --disable-radix-cache
```

```bash
python -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --dataset-name image \
    --num-prompts 500 \
    --image-count 3 \
    --image-resolution 720p \
    --random-input-len 512 \
    --random-output-len 512
```

2. To benchmark random dataset with 3000 prompts, 1024 input length, and 1024 output length, you can run:

**中文对照**：要对 random 数据集进行基准测试，3000 个提示词，1024 输入长度和 1024 输出长度，可以运行：

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-3B-Instruct
```

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random \
    --num-prompts 3000 \
    --random-input 1024 \
    --random-output 1024 \
    --random-range-ratio 0.5
```

### Choosing model and tokenizer

- `--model` is required unless the backend exposes `GET /v1/models`, in which case the first model ID is auto-selected.
- `--tokenizer` defaults to `--model`. Both can be HF model IDs or local paths.
- For ModelScope workflows, setting `SGLANG_USE_MODELSCOPE=true` enables fetching via ModelScope (weights are skipped for speed).
- If your tokenizer lacks a chat template, the script warns because token counting can be less robust for gibberish outputs.

**中文对照**：
### 选择模型和分词器

- `--model` 是必需的，除非后端暴露 `GET /v1/models`，在这种情况下，第一个模型 ID 将自动选择。
- `--tokenizer` 默认为 `--model`。两者都可以是 HF 模型 ID 或本地路径。
- 对于 ModelScope 工作流，设置 `SGLANG_USE_MODELSCOPE=true` 可以通过 ModelScope 获取（跳过权重以加快速度）。
- 如果你的分词器缺少聊天模板，脚本会发出警告，因为对于乱码输出，token 计数可能不太可靠。

### Rate, concurrency, and streaming

- `--request-rate`：每秒请求数。`inf` 立即发送所有请求（突发）。非无限速率使用泊松过程来处理到达时间。
- `--max-concurrency`：限制并发飞行中的请求，无论到达速率如何。
- `--disable-stream`：在支持时切换到非流式模式；对于聊天补全，TTFT 等于总延迟。

**中文对照**：
### 速率、并发和流式

- `--request-rate`：每秒请求数。`inf` 立即发送所有请求（突发）。非无限速率使用泊松过程来处理到达时间。
- `--max-concurrency`：限制并发飞行中的请求，无论到达速率如何。
- `--disable-stream`：在支持时切换到非流式模式；对于聊天补全，TTFT 等于总延迟。

### Other key options

- `--output-file FILE.jsonl`：将 JSONL 结果追加到文件；如果未指定则自动命名
- `--output-details`：包括每请求数组（生成的文本、错误、ttft、itl、输入/输出长度）
- `--extra-request-body '{"top_p":0.9,"temperature":0.6}'`：合并到载荷中（采样参数等）
- `--disable-ignore-eos`：传递 EOS 行为（因后端而异）
- `--warmup-requests N`：首先运行带有短输出的预热请求（默认 1）
- `--flush-cache`：在主运行前调用 `/flush_cache`（sglang）
- `--profile`：调用 `/start_profile` 和 `/stop_profile`（需要服务器启用分析，例如 `SGLANG_TORCH_PROFILER_DIR`）
- `--lora-name name1 name2 ...`：每个请求随机选择一个并传递给后端（例如 sglang 的 `lora_path`）
- `--tokenize-prompt`：发送整数 ID 而不是文本（目前仅支持 `--backend sglang`）

**中文对照**：
### 其他关键选项

- `--output-file FILE.jsonl`：将 JSONL 结果追加到文件；如果未指定则自动命名
- `--output-details`：包括每请求数组（生成的文本、错误、ttft、itl、输入/输出长度）
- `--extra-request-body '{"top_p":0.9,"temperature":0.6}'`：合并到载荷中（采样参数等）
- `--disable-ignore-eos`：传递 EOS 行为（因后端而异）
- `--warmup-requests N`：首先运行带有短输出的预热请求（默认 1）
- `--flush-cache`：在主运行前调用 `/flush_cache`（sglang）
- `--profile`：调用 `/start_profile` 和 `/stop_profile`（需要服务器启用分析，例如 `SGLANG_TORCH_PROFILER_DIR`）
- `--lora-name name1 name2 ...`：每个请求随机选择一个并传递给后端（例如 sglang 的 `lora_path`）
- `--tokenize-prompt`：发送整数 ID 而不是文本（目前仅支持 `--backend sglang`）

### Authentication

If your target endpoint requires OpenAI-style auth, set:

**中文对照**：
### 身份验证

如果目标端点需要 OpenAI 风格的身份验证，请设置：

```bash
export OPENAI_API_KEY=sk-...yourkey...
```

The script will add `Authorization: Bearer $OPENAI_API_KEY` automatically for OpenAI-compatible routes.

**中文对照**：脚本将自动为 OpenAI 兼容路由添加 `Authorization: Bearer $OPENAI_API_KEY`。

### Metrics explained

Printed after each run:

- Request throughput (req/s)
- Input token throughput (tok/s) - includes both text and vision tokens
- Output token throughput (tok/s)
- Total token throughput (tok/s) - includes both text and vision tokens
- Total input text tokens and Total input vision tokens - per-modality breakdown
- Concurrency: aggregate time of all requests divided by wall time
- End-to-End Latency (ms): mean/median/std/p99 per-request total latency
- Time to First Token (TTFT, ms): mean/median/std/p99 for streaming mode
- Inter-Token Latency (ITL, ms): mean/median/std/p95/p99/max between tokens
- TPOT (ms): Token processing time after first token, i.e., `(latency - ttft)/(tokens-1)`
- Accept length (sglang-only, if available): speculative decoding accept length

**中文对照**：
### 指标说明

每次运行后打印：

- 请求吞吐量（req/s）
- 输入 token 吞吐量（tok/s）- 包括文本和视觉 token
- 输出 token 吞吐量（tok/s）
- 总 token 吞吐量（tok/s）- 包括文本和视觉 token
- 总输入文本 token 和总输入视觉 token - 按模态细分
- 并发：所有请求的总时间除以挂钟时间
- 端到端延迟（ms）：每请求总延迟的均值/中位数/标准差/p99
- 首 token 时间（TTFT，ms）：流式模式的均值/中位数/标准差/p99
- Token 间延迟（ITL，ms）：token 之间的均值/中位数/标准差/p95/p99/最大值
- TPOT（ms）：首 token 后的 token 处理时间，即 `(latency - ttft)/(tokens-1)`
- 接受长度（仅 sglang，如果有）：推测解码接受长度

The script also retokenizes generated text with the configured tokenizer and reports "retokenized" counts.

**中文对照**：脚本还会使用配置的分词器对生成的文本进行重新分词，并报告"重新分词"计数。

### JSONL output format

When `--output-file` is set, one JSON object is appended per run. Base fields:

**中文对照**：
### JSONL 输出格式

当设置 `--output-file` 时，每次运行追加一个 JSON 对象。基础字段：

- Arguments summary: backend, dataset, request_rate, max_concurrency, etc.
- Duration and totals: completed, total_input_tokens, total_output_tokens, retokenized totals
- Throughputs and latency statistics as printed in the console
- `accept_length` when available (sglang)

**中文对照**：
- 参数摘要：backend、dataset、request_rate、max_concurrency 等
- 持续时间和总计：completed、total_input_tokens、total_output_tokens、retokenized 总计
- 控制台打印的吞吐量和延迟统计
- `accept_length`（当可用时，sglang）

With `--output-details`, an extended object also includes arrays:

- `input_lens`, `output_lens`
- `ttfts`, `itls` (per request: ITL arrays)
- `generated_texts`, `errors`

**中文对照**：使用 `--output-details` 时，扩展对象还包括数组：
- `input_lens`、`output_lens`
- `ttfts`、`itls`（每请求：ITL 数组）
- `generated_texts`、`errors`

### End-to-end examples

1) sglang native `/generate` (streaming):

**中文对照**：
### 端到端示例

1) sglang 原生 `/generate`（流式）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.5 \
  --num-prompts 2000 \
  --request-rate 100 \
  --max-concurrency 512 \
  --output-file sglang_random.jsonl --output-details
```

2) OpenAI-compatible Completions (e.g., vLLM):

**中文对照**：2) OpenAI 兼容补全（例如 vLLM）：

```bash
python3 -m sglang.bench_serving \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --sharegpt-output-len 256
```

3) OpenAI-compatible Chat Completions (streaming):

**中文对照**：3) OpenAI 兼容聊天补全（流式）：

```bash
python3 -m sglang.bench_serving \
  --backend vllm-chat \
  --base-url http://127.0.0.1:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --num-prompts 500 \
  --apply-chat-template
```

4) Images (VLM) with chat template:

**中文对照**：4) 带聊天模板的图像（VLM）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 2 \
  --image-resolution 720p \
  --random-input-len 128 --random-output-len 256 \
  --num-prompts 200 \
  --apply-chat-template
```

4a) Images with custom resolution:

**中文对照**：4a) 自定义分辨率的图像：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 1 \
  --image-resolution 512x768 \
  --random-input-len 64 --random-output-len 128 \
  --num-prompts 100 \
  --apply-chat-template
```

4b) 1080p images with PNG format and blank content:

**中文对照**：4b) 1080p PNG 格式空白内容图像：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 1 \
  --image-resolution 1080p \
  --image-format png \
  --image-content blank \
  --random-input-len 64 --random-output-len 128 \
  --num-prompts 100 \
  --apply-chat-template
```

5) Generated shared prefix (long system prompts + short questions):

**中文对照**：5) 生成的共享前缀（长系统提示 + 短问题）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 64 --gsp-prompts-per-group 16 \
  --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 256 \
  --num-prompts 1024
```

6) Tokenized prompts (ids) for strict length control (sglang only):

**中文对照**：6) 用于严格长度控制的分词提示词（id，仅 sglang）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --tokenize-prompt \
  --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.2
```

7) Profiling and cache flush (sglang):

**中文对照**：7) 分析和缓存刷新（sglang）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --profile \
  --flush-cache
```

8) TensorRT-LLM streaming endpoint:

**中文对照**：8) TensorRT-LLM 流式端点：

```bash
python3 -m sglang.bench_serving \
  --backend trt \
  --base-url http://127.0.0.1:8000 \
  --model your-trt-llm-model \
  --dataset-name random \
  --num-prompts 100 \
  --disable-ignore-eos
```

9) Evaluating large-scale KVCache sharing with mooncake trace (sglang only):

**中文对照**：9) 使用 mooncake 跟踪评估大规模 KVCache 共享（仅 sglang）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model model-name \
  --dataset-name mooncake \
  --mooncake-slowdown-factor 1.0 \
  --mooncake-num-rounds 1000 \
  --mooncake-workload conversation|mooncake|agent|synthetic
  --use-trace-timestamps true \
  --random-output-len 256
```

### Troubleshooting

- All requests failed: verify `--backend`, server URL/port, `--model`, and authentication. Check warmup errors printed by the script.
- Throughput seems too low: adjust `--request-rate` and `--max-concurrency`; verify server batch size/scheduling; ensure streaming is enabled if appropriate.
- Token counts look odd: prefer chat/instruct models with proper chat templates; otherwise tokenization of gibberish may be inconsistent.
- Image/MMMU datasets: ensure you installed extra deps (`pillow`, `datasets`, `pybase64`).
- Authentication errors (401/403): set `OPENAI_API_KEY` or disable auth on your server.

**中文对照**：
### 故障排除

- 所有请求失败：验证 `--backend`、服务器 URL/端口、`--model` 和身份验证。检查脚本打印的预热错误。
- 吞吐量似乎太低：调整 `--request-rate` 和 `--max-concurrency`；验证服务器批次大小/调度；确保在适当时启用了流式。
- Token 计数看起来奇怪：首选带有正确聊天模板的 chat/instruct 模型；否则对乱码的分词可能不一致。
- 图像/MMMU 数据集：确保安装了额外依赖项（`pillow`、`datasets`、`pybase64`）。
- 身份验证错误（401/403）：设置 `OPENAI_API_KEY` 或禁用服务器上的身份验证。

### Notes

- The script raises the file descriptor soft limit (`RLIMIT_NOFILE`) to help with many concurrent connections.
- For sglang, `/get_server_info` is queried post-run to report speculative decoding accept length when available.

**中文对照**：
### 注意事项

- 脚本会提高文件描述符软限制（`RLIMIT_NOFILE`）以帮助处理许多并发连接。
- 对于 sglang，在运行后查询 `/get_server_info` 以报告推测解码接受长度（如果可用）。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/bench_serving.py` | 主基准测试脚本：异步请求生成、指标收集（TTFT、ITL、吞吐量、延迟） |
| `python/sglang/srt/entrypoints/http_server.py` | 提供 `/generate`、`/v1/completions`、`/v1/chat/completions` 端点供 bench_serving 测试 |
| `python/sglang/srt/managers/scheduler_profiler_mixin.py` | 服务器端分析器控制，通过 `--profile` 标志触发 `/start_profile` 和 `/end_profile` |

### 关键代码逻辑

- **后端分发**：`bench_serving.py` 根据 `--backend`（sglang/vllm/trt/lmdeploy）选择请求格式，映射到相应端点和载荷模式
- **指标管道**：异步 `aiohttp` 客户端收集每请求的 TTFT、ITL 数组和端到端延迟；汇总为平均值/中位数/p95/p99 统计数据
- **数据集生成**：支持 sharegpt（真实对话）、random（合成）、image（VLM）、generated-shared-prefix（前缀缓存）、mooncake（KV 共享跟踪）

### 集成要点

- **速率控制**：`--request-rate`（泊松到达）、`--max-concurrency`（飞行中上限）
- **输出**：`--output-file` 生成包含完整指标的 JSONL；`--output-details` 包含每请求数组
- **分析**：`--profile` 调用 sglang 服务器的 `/start_profile` 和 `/stop_profile`；需要 `SGLANG_TORCH_PROFILER_DIR`
