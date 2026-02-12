# GLM-4.6V / GLM-4.5V Usage

**中文对照**：GLM-4.6V / GLM-4.5V 使用指南

## Launch commands for SGLang

**中文对照**：SGLang 启动命令

Below are suggested launch commands tailored for different hardware / precision modes

**中文对照**：以下是针对不同硬件/精度模式的建议启动命令

### FP8 (quantised) mode

**中文对照**：FP8（量化）模式

For high memory-efficiency and latency optimized deployments (e.g., on H100, H200) where FP8 checkpoint is supported:

**中文对照**：对于支持 FP8 检查点的高内存效率和延迟优化部署（例如在 H100、H200 上）：

```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V-FP8 \
  --tp 2 \
  --ep 2 \
  --host 0.0.0.0 \
  --port 30000 \
  --keep-mm-feature-on-device
```

### Non-FP8 (BF16 / full precision) mode

**中文对照**：非 FP8（BF16 / 全精度）模式

For deployments on A100/H100 where BF16 is used (or FP8 snapshot not used):

**中文对照**：对于在 A100/H100 上使用 BF16（或不使用 FP8 快照）的部署：
```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V \
  --tp 4 \
  --ep 4 \
  --host 0.0.0.0 \
  --port 30000
```

## Hardware-specific notes / recommendations

**中文对照**：硬件特定说明/建议

- On H100 with FP8: Use the FP8 checkpoint for best memory efficiency.

**中文对照**：- 在 H100 上使用 FP8：使用 FP8 检查点以获得最佳内存效率。

- On A100 / H100 with BF16 (non-FP8): It's recommended to use `--mm-max-concurrent-calls` to control parallel throughput and GPU memory usage during image/video inference.

**中文对照**：- 在 A100 / H100 上使用 BF16（非 FP8）：建议使用 `--mm-max-concurrent-calls` 来控制图像/视频推理期间的并行吞吐量和 GPU 内存使用。

- On H200 & B200: The model can be run "out of the box", supporting full context length plus concurrent image + video processing.

**中文对照**：- 在 H200 和 B200 上：该模型可以直接运行，支持完整上下文长度以及并发图像+视频处理。

## Sending Image/Video Requests

**中文对照**：发送图像/视频请求

### Image input:

**中文对照**：图像输入：

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "zai-org/GLM-4.6V",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print(response.text)
```

### Video Input:

**中文对照**：视频输入：

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "zai-org/GLM-4.6V",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's happening in this video?"},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/jobs_presenting_ipod.mp4"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print(response.text)
```

## Important Server Parameters and Flags

**中文对照**：重要的服务器参数和标志

When launching the model server for **multimodal support**, you can use the following command-line arguments to fine-tune performance and behavior:

**中文对照**：在为**多模态支持**启动模型服务器时，可以使用以下命令行参数来微调性能和行为：

- `--mm-attention-backend`: Specify multimodal attention backend. Eg. `fa3`(Flash Attention 3)

**中文对照**：- `--mm-attention-backend`：指定多模态注意力后端。例如 `fa3`（Flash Attention 3）

- `--mm-max-concurrent-calls <value>`: Specifies the **maximum number of concurrent asynchronous multimodal data processing calls** allowed on the server. Use this to control parallel throughput and GPU memory usage during image/video inference.

**中文对照**：- `--mm-max-concurrent-calls <value>`：指定服务器允许的**最大并发异步多模态数据处理调用数**。使用此参数来控制图像/视频推理期间的并行吞吐量和 GPU 内存使用。

- `--mm-per-request-timeout <seconds>`: Defines the **timeout duration (in seconds)** for each multimodal request. If a request exceeds this time limit (e.g., for very large video inputs), it will be automatically terminated.

**中文对照**：- `--mm-per-request-timeout <seconds>`：定义每个多模态请求的**超时持续时间（秒）**。如果请求超过此时间限制（例如，对于非常大的视频输入），它将自动终止。

- `--keep-mm-feature-on-device`: Instructs the server to **retain multimodal feature tensors on the GPU** after processing. This avoids device-to-host (D2H) memory copies and improves performance for repeated or high-frequency inference workloads.

**中文对照**：- `--keep-mm-feature-on-device`：指示服务器在处理后将**多模态特征张量保留在 GPU 上**。这避免了设备到主机（D2H）内存复制，并提高了重复或高频推理工作负载的性能。

- `--mm-enable-dp-encoder`: Placing the ViT in data parallel while keeping the LLM in tensor parallel consistently lowers TTFT and boosts end-to-end throughput.

**中文对照**：- `--mm-enable-dp-encoder`：将 ViT 置于数据并行同时将 LLM 保持张量并行，可以持续降低 TTFT 并提升端到端吞吐量。

- `SGLANG_USE_CUDA_IPC_TRANSPORT=1`: Shared memory pool based CUDA IPC for multi-modal data transport. For significantly improving e2e latency.

**中文对照**：- `SGLANG_USE_CUDA_IPC_TRANSPORT=1`：基于共享内存池的 CUDA IPC 用于多模态数据传输。可以显著改善端到端延迟。

### Example usage with the above optimizations:
```bash
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_VLM_CACHE_SIZE_MB=0 \
python -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --tp-size 8 \
  --enable-cache-report \
  --log-level info \
  --max-running-requests 64 \
  --mem-fraction-static 0.65 \
  --chunked-prefill-size 8192 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --mm-enable-dp-encoder \
  --enable-metrics
```

### Thinking Budget for GLM-4.5V / GLM-4.6V

**中文对照**：GLM-4.5V / GLM-4.6V 的思考预算

In SGLang, we can implement thinking budget with `CustomLogitProcessor`.

**中文对照**：在 SGLang 中，我们可以使用 `CustomLogitProcessor` 实现思考预算。

Launch a server with the `--enable-custom-logit-processor` flag. Then, use `Glm4MoeThinkingBudgetLogitProcessor` in the request, similar to the `GLM-4.6` example in [glm45.md](./glm45.md).

**中文对照**：使用 `--enable-custom-logit-processor` 标志启动服务器。然后在请求中使用 `Glm4MoeThinkingBudgetLogitProcessor`，类似于 [glm45.md](./glm45.md) 中的 `GLM-4.6` 示例。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/glm4v.py` | GLM-4.6V 多模态模型架构：视觉编码器 + 语言解码器 |
| `python/sglang/srt/multimodal/mm_utils.py` | 多模态输入的图像/视频预处理 |
| `python/sglang/srt/multimodal/vit_cuda_graph_runner.py` | 视觉编码器的 ViT CUDA Graph 优化 |

### 集成要点

- **服务器参数**：`--tp`、`--ep`、`--keep-mm-feature-on-device`、`--mm-attention-backend fa3`、`--mm-max-concurrent-calls`、`--mm-enable-dp-encoder`
- **环境变量**：`SGLANG_USE_CUDA_IPC_TRANSPORT=1` 用于 CUDA IPC 多模态传输，`SGLANG_VLM_CACHE_SIZE_MB` 用于 VLM 缓存控制
- **API**：兼容 OpenAI 的 `/v1/chat/completions` 端点，支持 `image_url` 和 `video_url` 内容类型
