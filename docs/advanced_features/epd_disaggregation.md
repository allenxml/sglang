# EPD Disaggregation

## Why and What is EPD Disaggregation?

In modern Vision-Language Model (VLM) inference, request execution naturally decomposes into three distinct stages: Encoder, Prefill, and Decode.
The Encoder stage performs vision preprocessing and ViT-based image encoding, which is highly compute-intensive but only required during request initialization. The Prefill stage processes the full multimodal input sequence to initialize the language model's Key-Value (KV) cache, while the Decode stage is dominated by memory bandwidth and KV cache access for autoregressive token generation.

**中文对照**：## EPD 分离的原因和定义

在现代视觉语言模型（VLM）推理中，请求执行自然分解为三个不同的阶段：编码器、预填充和解码。编码器阶段执行视觉预处理和基于 ViT 的图像编码，这是计算密集型的，但仅在请求初始化期间需要。预填充阶段处理完整的多模态输入序列以初始化语言模型的键值（KV）缓存，而解码阶段主要由内存带宽和用于自回归令牌生成的 KV 缓存访问主导。

Existing deployments typically colocate these stages within a unified execution engine, or at best apply Prefill–Decode (PD) disaggregation. However, such designs still tightly couple vision encoding with language prefill, leading to inefficient resource utilization, limited scalability for image-heavy workloads, and suboptimal scheduling under load.

**中文对照**：现有的部署通常将这些阶段共置在统一的执行引擎中，或者最多应用预填充-解码（PD）分离。然而，这样的设计仍然将视觉编码与语言预填充紧密耦合，导致资源利用效率低下、图像繁重工作负载的可扩展性有限以及负载下的次优调度。

To address these challenges, we introduce Encoder–Prefill–Decode (EPD) Disaggregation in SGLang. EPD further separates vision encoding from language processing, enabling independent horizontal scaling of encoder servers, improved load balancing for multimodal requests, and seamless integration with existing PD disaggregation to form a fully decoupled three-tier inference architecture.

**中文对照**：为了应对这些挑战，我们在 SGLang 中引入了编码器-预填充-解码（EPD）分离。EPD 进一步将视觉编码与语言处理分离，支持编码器服务器的独立水平扩展，改善多模态请求的负载平衡，并与现有的 PD 分离无缝集成，形成完全解耦的三层推理架构。

### Usage

You can launch a language-only model using `--language-only`, or an encoder-only model using `--encoder-only`.
When launching a language-only model, you must additionally specify the encoder service endpoints via `--encoder-urls`.

We support multiple encoder transfer backends, including zmq_to_scheduler, zmq_to_tokenizer, and mooncake (the default is zmq_to_scheduler). The backend can be selected using `--encoder-transfer-backend`.

**中文对照**：### 使用方法

您可以使用 `--language-only` 启动仅语言模型，或使用 `--encoder-only` 启动仅编码器模型。启动仅语言模型时，您必须通过 `--encoder-urls` 额外指定编码器服务终端。

我们支持多种编码器传输后端，包括 zmq_to_scheduler、zmq_to_tokenizer 和 mooncake（默认为 zmq_to_scheduler）。可以使用 `--encoder-transfer-backend` 选择后端。

#### Qwen VL

- EP Disaggregation

```bash
# encoder 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000
# encoder 1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
# language-only server
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
```

- EPD Disaggregation

```bash
# encoder 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000
# encoder 1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
# prefill 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode prefill \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
# decode 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode decode \
  --port 30003
# router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://$PREFILL_HOST:30002 \
  --decode http://$DECODE_HOST:30003 \
  --port 8000

```

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/server_args.py` | 定义 `--encoder-only`、`--language-only`、`--encoder-urls`、`--encoder-transfer-backend` 标志 |
| `python/sglang/srt/configs/model_config.py` | 检测encoder-only / language-only模式并调整模型架构配置 |
| `python/sglang/srt/model_executor/model_runner.py` | 条件前向传递：根据模式仅运行编码器或仅运行语言模型 |
| `python/sglang/srt/managers/tokenizer_manager.py` | 通过ZMQ或Mooncake传输将编码器embeddings路由到语言服务器 |
| `python/sglang/srt/managers/scheduler.py` | 处理带有预计算编码器输出的请求，对 `--language-only` 跳过ViT阶段 |

### 架构

```
[Client Request (Image + Text)]
        │
        ▼
[Encoder Server] ──encoder-transfer-backend──▶ [Prefill Server] ──PD──▶ [Decode Server]
  (--encoder-only)       (zmq/mooncake)        (--language-only         (--disaggregation
   ViT encoding                                  --disaggregation          -mode decode)
                                                  -mode prefill)
        │                                              │                       │
        ▼                                              ▼                       ▼
  Encoder embeddings                          KV cache init              Autoregressive
  sent to language server                     + KV transfer              token generation
```

### 关键代码逻辑

- **模式检测**: `server_args.py` 解析 `--encoder-only` 和 `--language-only` 标志；`model_config.py` 使用这些标志将模型拆分为仅编码器或仅解码器组件
- **编码器传输**: `tokenizer_manager.py` 实现基于ZMQ（`zmq_to_scheduler`、`zmq_to_tokenizer`）和基于Mooncake的传输，用于将编码器embeddings发送到语言服务器
- **三层流水线**: 将EP分离（编码器分离）与现有PD分离（prefill-decode分离）结合，形成完全解耦的E-P-D架构

### 集成要点

- **服务器标志**: `--encoder-only`、`--language-only`、`--encoder-urls`、`--encoder-transfer-backend`
- **兼容PD分离**: `--disaggregation-mode prefill/decode` 可与编码器分离配合使用
- **路由器**: `sglang_router.launch_router --pd-disaggregation` 处理prefill和decode服务器之间的路由
