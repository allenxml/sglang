# Hyperparameter Tuning

## Achieving high throughput for offline batch inference

Achieving a large batch size is the most important thing for attaining high throughput in offline batch inference.
When the server is running at full load in a steady state, look for the following in the log:

```Decode batch. #running-req: 233, #token: 370959, token usage: 0.82, cuda graph: True, gen throughput (token/s): 4594.01, #queue-req: 317```

### Adjust the request submission speed to control `#queue-req`

`#queue-req` indicates the number of requests in the queue.
If you frequently see `#queue-req: 0`, it suggests that your client code is submitting requests too slowly.
A healthy range for `#queue-req` is `100 - 2000`.
However, avoid making `#queue-req` too large, as this will increase the scheduling overhead on the server.

**中文对照**：# 超参数调优

## 实现离线批处理推理的高吞吐量

实现大批量是在离线批处理推理中获得高吞吐量的最重要的事情。当服务器在稳态下满负载运行时，请在日志中查找以下内容：

```Decode batch. #running-req: 233, #token: 370959, token usage: 0.82, cuda graph: True, gen throughput (token/s): 4594.01, #queue-req: 317```

### 调整请求提交速度以控制 `#queue-req`

`#queue-req` 表示队列中的请求数量。如果您经常看到 `#queue-req: 0`，这表明您的客户端代码提交请求太慢了。`#queue-req` 的健康范围是 `100 - 2000`。但是，避免使 `#queue-req` 过大，因为这会增加服务器上的调度开销。

### Achieve a high `token usage`

`token usage` indicates the KV cache memory utilization of the server. `token usage > 0.9` means good utilization.

If you frequently see `token usage < 0.9` and `#queue-req > 0`, it means the server is too conservative about taking in new requests. You can decrease `--schedule-conservativeness` to a value like 0.3.
The case of a server being too conservative can happen when users send many requests with a large `max_new_tokens` but the requests stop very early due to EOS or stop strings.

On the other hand, if you see `token usage` very high and you frequently see warnings like
`KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_token_ratio: 0.9998 -> 1.0000`, you can increase `--schedule-conservativeness` to a value like 1.3.
If you see `KV cache pool is full. Retract requests.` occasionally but not frequently (~1 time per minute), it is okay.

**中文对照**：### 实现高 `token usage`

`token usage` 表示服务器的 KV 缓存内存利用率。`token usage > 0.9` 表示良好的利用率。

如果您经常看到 `token usage < 0.9` 且 `#queue-req > 0`，这意味着服务器在接受新请求方面过于保守。您可以将 `--schedule-conservativeness` 降低到类似 0.3 的值。服务器过于保守的情况可能发生在用户发送许多具有较大 `max_new_tokens` 的请求，但由于 EOS 或停止字符串导致请求很早就停止时。

另一方面，如果您看到 `token usage` 非常高，并且经常看到类似以下警告
`KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_token_ratio: 0.9998 -> 1.0000`，您可以将 `--schedule-conservativeness` 增加到类似 1.3 的值。如果您偶尔（大约每分钟一次）看到 `KV cache pool is full. Retract requests.`，这是可以的。

### Tune `--mem-fraction-static` to increase KV cache pool capacity
SGLang allocates memory as follows:

Total memory usage = model weights + KV cache pool + CUDA graph buffers + activations

The `--mem-fraction-static` parameter determines how much memory is allocated to the first two components:

mem_fraction_static = (model weights + KV cache pool) / GPU memory capacity

To support higher concurrency, you should maximize the KV cache pool capacity by setting `--mem-fraction-static` as high as possible while still reserving enough memory for activations and CUDA graph buffers.

SGLang uses simple heuristics to set the default value of `--mem-fraction-static`, but you can optimize it for your use cases.
As a rule of thumb, reserving 5–8 GB of memory for activations is typically sufficient. You can check this by inspecting the logs just before the server is ready.
Look for log entries like this:

```
[2025-08-11 17:17:03] max_total_num_tokens=665690, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=65536, available_gpu_mem=13.50 GB
```

Check the `available_gpu_mem` value.
- If it is between 5–8 GB, the setting is good.
- If it is too high (e.g., 10 - 20 GB), increase `--mem-fraction-static` to allocate more memory to the KV cache.
- If it is too low, you risk out-of-memory (OOM) errors later, so decrease `--mem-fraction-static`.

Another straightforward approach is to increase `--mem-fraction-static` in increments of 0.01 until you encounter OOM errors for your workloads.

**中文对照**：### 调优 `--mem-fraction-static` 以增加 KV 缓存池容量
SGLang 按以下方式分配内存：

总内存使用量 = 模型权重 + KV 缓存池 + CUDA graph 缓冲区 + 激活值

`--mem-fraction-static` 参数决定分配给前两个组件的内存量：

mem_fraction_static = (模型权重 + KV 缓存池) / GPU 内存容量

为了支持更高的并发，您应该通过将 `--mem-fraction-static` 设置得尽可能高来最大化 KV 缓存池容量，同时仍为激活值和 CUDA graph 缓冲区保留足够的内存。

SGLang 使用简单的启发式方法来设置 `--mem-fraction-static` 的默认值，但您可以根据自己的用例进行优化。根据经验，为激活值保留 5-8 GB 的内存通常就足够了。您可以通过在服务器准备就绪之前检查日志来确认这一点。查找类似以下的日志条目：

```
[2025-08-11 17:17:03] max_total_num_tokens=665690, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=65536, available_gpu_mem=13.50 GB
```

检查 `available_gpu_mem` 值。
- 如果在 5-8 GB 之间，设置是好的。
- 如果太高（例如 10-20 GB），增加 `--mem-fraction-static` 以将更多内存分配给 KV 缓存。
- 如果太低，您以后可能会遇到内存不足（OOM）错误，因此请降低 `--mem-fraction-static`。

另一种直接的方法是以 0.01 的增量增加 `--mem-fraction-static`，直到您的工作负载遇到 OOM 错误。

### Avoid out-of-memory errors by tuning `--chunked-prefill-size`, `--mem-fraction-static`, and `--max-running-requests`

If you encounter out-of-memory (OOM) errors, you can adjust the following parameters:

- If OOM occurs during prefill, try reducing `--chunked-prefill-size` to `4096` or `2048`. This saves memory but slows down the prefill speed for long prompts.
- If OOM occurs during decoding, try lowering `--max-running-requests`.
- You can also reduce `--mem-fraction-static` to a smaller value, such as 0.8 or 0.7. This decreases the memory usage of the KV cache memory pool and helps prevent OOM errors during both prefill and decoding. However, it limits maximum concurrency and reduces peak throughput.

### Tune `--cuda-graph-max-bs`
By default, CUDA graph is enabled only for small batch sizes (e.g., less than 160 or 256).
However, for some models, especially at large tensor parallelism sizes, CUDA graph can be useful for batch sizes up to 512 or 768.
Therefore, it may be beneficial to increase `--cuda-graph-max-bs` to a larger value.
Note that CUDA graph consumes more memory, so you may need to reduce `--mem-fraction-static` at the same time.

### Tune `--dp-size` and `--tp-size`

Data parallelism is better for throughput. When there is enough GPU memory, always favor data parallelism for throughput. Refer to [SGLang Model Gateway (former Router)](../advanced_features/sgl_model_gateway.md) for a better data parallelism rather than using `dp_size` parameter.

### Try other options

- `torch.compile` accelerates small models on small batch sizes. You can enable it with `--enable-torch-compile`.
- Try other quantization (e.g. FP8 quantization with `--quantization fp8`)
- Try other parallelism strategies (e.g. [expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/)) or DP attention for deepseek models (with `--enable-dp-attention --dp-size 8`).
- If the workload has many shared prefixes, try `--schedule-policy lpm`. Here, `lpm` stands for longest prefix match. It reorders requests to encourage more cache hits but introduces more scheduling overhead.

**中文对照**：### 通过调优 `--chunked-prefill-size`、`--mem-fraction-static` 和 `--max-running-requests` 来避免内存不足错误

如果您遇到内存不足（OOM）错误，可以调整以下参数：

- 如果 OOM 发生在预填充期间，尝试将 `--chunked-prefill-size` 减少到 `4096` 或 `2048`。这可以节省内存，但会减慢长提示的预填充速度。
- 如果 OOM 发生在解码期间，尝试降低 `--max-running-requests`。
- 您还可以将 `--mem-fraction-static` 降低到较小的值，例如 0.8 或 0.7。这会减少 KV 缓存内存池的内存使用，并有助于防止预填充和解码期间的 OOM 错误。但是，它会限制最大并发并降低峰值吞吐量。

### 调优 `--cuda-graph-max-bs`
默认情况下，CUDA graph 仅对小批量大小启用（例如小于 160 或 256）。
然而，对于某些模型，特别是在较大的张量并行大小下，CUDA graph 对于高达 512 或 768 的批量大小可能有用。
因此，将 `--cuda-graph-max-bs` 增加到一个更大的值可能是有益的。
请注意，CUDA graph 会消耗更多内存，因此您可能需要同时减少 `--mem-fraction-static`。

### 调优 `--dp-size` 和 `--tp-size`

数据并行更适合吞吐量。当有足够的 GPU 内存时，始终优先使用数据并行来提高吞吐量。请参阅 [SGLang 模型网关（前路由器）](../advanced_features/sgl_model_gateway.md) 以获得更好的数据并行，而不是使用 `dp_size` 参数。

### 尝试其他选项

- `torch.compile` 可以加速小模型在小批量大小上的推理。您可以使用 `--enable-torch-compile` 来启用它。
- 尝试其他量化（例如使用 `--quantization fp8` 的 FP8 量化）
- 尝试其他并行策略（例如[专家并行](https://lmsys.org/blog/2025-05-05-large-scale-ep/)）或 deepseek 模型的 DP 注意力（使用 `--enable-dp-attention --dp-size 8`）。
- 如果工作负载有许多共享前缀，请尝试 `--schedule-policy lpm`。这里，`lpm` 代表最长前缀匹配。它重新排序请求以鼓励更多缓存命中，但会增加调度开销。

## 代码实现
- **核心文件**: `python/sglang/srt/server_args.py`
- **架构**: 超参数通过 `ServerArgs` 数据类及其关联的 `add_cli_args` 方法管理。这集中了 SGLang 运行时使用的参数定义、默认值和文档字符串。
- **关键代码片段**:
  - `parser.add_argument("--schedule-conservativeness", ...)`: 控制基于 KV 缓存利用率接纳新请求的激进程度。
  - `parser.add_argument("--mem-fraction-static", ...)`: 定义相对于总 GPU 内存的模型权重和 KV 缓存池的分配。
  - `parser.add_argument("--chunked-prefill-size", ...)`: 配置分割大型预填充操作以防止 OOM 错误的 token 阈值。
- **集成要点**: 这些参数在服务器启动时解析并传播到核心组件，如 `Scheduler` 和 `ModelRunner`。它们可以通过 CLI 标志或配置文件调整，以针对特定硬件和工作负载优化性能。
