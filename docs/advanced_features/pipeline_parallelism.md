# Pipeline Parallelism for Long Context

## Why Pipeline Parallelism?

As Large Language Models (LLMs) scale toward trillion-parameter architectures and "infinite" context windows, the underlying serving infrastructure must evolve toward more granular, cross-node parallelization strategies. While KV cache techniques effectively mitigate redundant computation, they cannot circumvent the prohibitive Time to First Token (TTFT) inherent in ultra-long sequences with extremely large initial Input Token Length (ITL). Although Tensor Parallelism (TP) remains the conventional approach for intra-node scaling, it frequently encounters communication bottlenecks during multi-node deployments. On the other hand, pipeline parallelism only requires cross-node communication at the boundaries of each pipeline stage, which can achieve better computation-communication overlap compared to a large TP. Therefore, it is also a promising parallelization strategy for improving throughput.

Detailed analysis can be found in this [blog](https://lmsys.org/blog/2026-01-15-chunked-pipeline/).

**中文对照**：# 用于长上下文的流水线并行

## 为什么需要流水线并行？

随着大语言模型（LLM）扩展到万亿参数架构和"无限"上下文窗口，底层服务基础设施必须向更细粒度、跨节点的并行化策略发展。虽然 KV 缓存技术有效减轻了冗余计算，但它们无法规避超长序列中固有的首批令牌时间（TTFT）问题，这些序列具有极大的初始输入令牌长度（ITL）。虽然张量并行（TP）仍然是节点内扩展的常规方法，但在多节点部署中经常遇到通信瓶颈。另一方面，流水线并行只需要在每个流水线阶段的边界进行跨节点通信，与大型 TP 相比可以实现更好的计算-通信重叠。因此，它也是提高吞吐量的一个有前途的并行化策略。

详细的分析可以在这篇[博客](https://lmsys.org/blog/2026-01-15-chunked-pipeline/)中找到。

## 基于异步通信的实现重构

通过动态分块预填充，流水线并行有潜力减少长上下文输入的 TTFT。对于每个请求，其输入令牌可以分成多个块，每个块不大于分块预填充大小。同一请求的不同块可以同时由不同的节点处理，从而并行化处理并减少 TTFT。SGLang 已经支持流水线并行（#5724）一段时间，并使其与 PD 分离功能（#8846）兼容，但实现并不完美，有很大的性能改进空间。

To eliminate this performance hazard, SGLang implements a Micro-batching Event Loop with non-blocking asynchronous peer-to-peer (P2P) communication to overlap GPU computation with CPU metadata processing and PP communication. This ensures that while one micro-batch is being computed on the GPU, the next one is already being prepared and moved into position effectively, ensuring the pipeline remains as saturated as possible. This approach was first proposed in #7979 and has been redesigned and included in #11852.

The key mechanisms of the implementation include:

* **Decoupled Sync/Async Logic in the Event Loop:** The scheduler uses `async_send` in `_pp_send_pyobj_to_next_stage`. Instead of waiting for a transfer to complete, it returns a `P2PWork` handle. The actual synchronization (`P2PWork.work.wait()`) is deferred until `_pp_commit_comm_work` is called, allowing the CPU to perform other work—like scheduling the next batch or processing metadata—while data is in flight.
* **Multi-Stream Execution:** In addition to the main `default_stream`, which serves as the synchronization stream, SGLang utilizes dedicated `forward_stream` and `copy_stream` to execute forward pass GPU computation and Data-to-Host (D2H) memory transfers separately for better overlapping. While `_pp_launch_batch` is executing the current micro-batch on the GPU for the current stage, the CPU processes the previous micro-batch's results using `_pp_process_batch_result`.

**中文对照**：为了消除这种性能隐患，SGLang 实现了微批次事件循环，使用非阻塞异步点对点（P2P）通信来重叠 GPU 计算与 CPU 元数据处理和 PP 通信。这确保了当一个微批次正在 GPU 上计算时，下一个已经准备好并有效地移动到位，确保流水线保持尽可能饱和。该方法最初在 #7979 中提出，并在 #11852 中重新设计和包含。

实现的关键机制包括：

* **事件循环中解耦的同步/异步逻辑：** 调度器在 `_pp_send_pyobj_to_next_stage` 中使用 `async_send`。它不等待传输完成，而是返回一个 `P2PWork` 句柄。实际的同步（`P2PWork.work.wait()`）延迟到调用 `_pp_commit_comm_work` 时，允许 CPU 在数据传输过程中执行其他工作——比如调度下一个批次或处理元数据。
* **多流执行：** 除了作为同步流的主 `default_stream` 之外，SGLang 还利用专用的 `forward_stream` 和 `copy_stream` 分别执行前向传递 GPU 计算和数据到主机（D2H）内存传输，以实现更好的重叠。当 `_pp_launch_batch` 在当前阶段的 GPU 上执行当前微批次时，CPU 使用 `_pp_process_batch_result` 处理前一个微批次的结果。

## Guidance about Dynamic Chunking

### Why Dynamic Chunking
Chunked prefill with a fixed size can cause bubbles in the pipeline, especially when the pp size is large. The main reason behind this phenomenon is that the model has a non-uniform running time, even though each chunk size is identical (brought by the Transformer structure). The larger the prefix sequence length, the longer the running time of the chunk. And these bubbles will be propagated to the next stage, and will significantly degrade the scale efficiency of larger pp ranks.

To address this issue, SGLang introduces a dynamic chunking mechanism to predict the optimal size for the next chunk such that it satisfies this condition:

Runtime(L + Next Chunk Size) - Runtime(L) = Runtime(Initial Chunk Size)

where ***L*** denotes the Prefix Sequence Length. By profiling a series of requests with different ITLs, we model the cumulative runtime as a quadratic function of sequence length. Using this model, we solve the optimal next chunk size for any given prefix length ***L***. Since the computation complexity of the Attention mechanism scales with ***L***, the next chunk size will be progressively reduced as ***L*** grows to maintain an aligned chunk execution time across pipeline stages.

Based on this method, the scheduler can predict and dynamically reduce the chunk size during runtime to minimize the bubbles caused by the stage misalignment. To be noticed, the scheduler does not use the raw predicted value. To facilitate efficient KVCache memory management and ensure affinity with hardware execution efficiency, the value is aligned downward to the nearest multiple of max(`--page-size`, 64).


### Chunked Prefill Size and Smoothing Factor

When `--enable-dynamic-chunking` is enabled, each chunk size of a sequence is determined dynamically based on the quadratic model that predicts the next chunk size based on the estimated runtime of the initial chunk length. In this case, we use `--chunked-prefill-size` to set up the initial chunk size. When switching to the dynamic chunking mode, the initial chunk size (`--chunked-prefill-size`) should be set to a larger value comparable to the original chunked prefill size, so that there won't be too many chunks.

**`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`** is an environmental variable that controls the smoothing factor for the dynamic chunking algorithm, defaulting to 0.75. It determines how much the chunk size can change during the prefill phase. A larger value means a more aggressive chunk size change, which may lead to better performance but also to greater chunk size changes (the chunk size at the end may become very small, which could lead to performance degradation) and more total chunks. When it is set to 1, the chunk size will be adjusted strictly based on the aforementioned quadratic model that predicts the next chunk size. A smaller value means a more conservative chunk size change, which may lead to smaller chunk size changes and fewer total chunks. When it is set to 0, the chunk size will not be adjusted dynamically, so it is identical to the traditional way with a fixed chunked prefill size.

Due to the variation in hardware, models, and target workloads, a static configuration is seldom optimal across all scenarios. Consequently, achieving peak performance necessitates a degree of hyperparameter tuning when switching to the dynamic chunking mode.

**Tuning Guidance for Dynamic Chunked Prefill**

* **Step 1 \- Iterate to find the optimal fixed chunked prefill size for the targeted PP size**: Different PP sizes for targeted ITL may have different optimal chunked prefill sizes. Therefore, users should iterate to obtain the baseline according to the available resources for scaling.
* **Step 2 \- Initial Chunk Size Selection for Dynamic Chunking**: Set the initial size to 2× or 3× the optimal fixed chunked prefill size. This reduces the total number of chunks and prevents "tail chunks" from underutilizing hardware. To maintain efficiency for extremely large Input Token Lengths (ITL), the dynamic predictor automatically ensures subsequent chunks are at least 1/4 of this initial size. In addition, it is recommended to use a larger initial chunk size (e.g., 4× the optimal fixed chunked prefill size) for such cases as well.
* **Step 3 \- Smooth Factor Adjustment**: This factor controls how strictly the chunk size adjusts the prediction given by the quadratic performance fitting model.
  * 1.0: Follows the model strictly.
  * **0.6 – 0.85 (Recommended)**: Typical range for the best balance between dynamic scaling and hardware stability. Through experiments, we find that a range between 0.6 and 0.85 typically yields the best performance for dynamic chunking.
  * 0: Disables dynamic adjustment, reverting to traditional fixed-size chunking.
* **Another small optimization tip:** Put the larger partition in the higher PP rank when the layers are not evenly divisible across ranks. It can increase the GPU utilization when a larger PP rank is waiting for the previous stage’s result, hence reducing the bubbles on higher PP ranks. If we take DeepSeek-V3.1 as an example, `SGLANG_PP_LAYER_PARTITION=15,15,15,16` usually performs better than `16,15,15,15`.

## Best Practice for Long Context

### Tuning the Chunked Prefill Size
Optimizing the chunked prefill size is crucial for balancing pipeline efficiency and resource utilization. The ideal size depends on factors including model architecture, hardware configuration, and typical input lengths. We recommend starting with a small chunk size, such as 4K, and gradually increasing it until you find the optimal size for your specific use case (Different targeted ITL and PP Sizes may have different optimal chunked prefill sizes. Therefore, users should iterate to obtain the baseline according to the available resources for scaling). Alternatively, you can analyze the hardware capacity and determine the optimal chunk size based on the roofline model.

### Enable Dynamic Chunking and Adjust Smoothing Factor for Ultra-long ITL
SGLang also offers a dynamic chunking solution that could further improve performance. This feature is currently an experimental feature that requires a certain amount of tuning experimentation and may not be suitable for all workloads. In addition, fine-tuning the smoothing factor can help optimize performance for specific workloads and model characteristics.

### Case Study on NVIDIA H20

When evaluating pipeline parallelism with fixed chunked prefill sizes from 2K to 16K, experiment results show that a 4K chunk size delivered optimal prefill TTFT performance for the DeepSeek-V3.1, and a 6K chunk size delivered optimal prefill TTFT performance for the Qwen3-235B-A22B-FP8.

When enabling dynamic chunking, we first scale the optimal fixed chunked prefill size by a factor of 3 as the initial chunk size. Through experimentation, we found that a multiplier of 2-3 provides an appropriate balance—avoiding excessive initial pipeline bubbles while ensuring that subsequent chunks don't become too small as context length increases. With the default dynamic chunking smoothing factor of 0.75, we performed parameter tuning and determined that a value of 0.65 works optimally with the 12K initial chunk size for the DeepSeek-V3.1, while a value of 0.8 works optimally with the 18K initial chunk size for the Qwen3-235B-A22B-FP8.

#### DeepSeek-V3.1 with 128K Input Token Length
```bash
# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 4096
```

```bash
# prefill node 0 (with dynamic chunking)
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 12288 --enable-dynamic-chunking
```

#### Qwen3-235B-A22B-FP8 with 128K Input Token Length
```bash
# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 6144
```

```bash
# prefill node 0 (with dynamic chunking)
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.8
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 18432 --enable-dynamic-chunking
```

Note: `--disable-radix-cache` is enabled only for reproducible benchmarking purposes. It is not recommended to use it in production.

## Best Practice for Pipeline Parallelism with PD Disaggregation
To be added. Stay tuned for the latest updates on Pipeline Parallelism with PD Disaggregation.

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/managers/scheduler_pp_mixin.py` | 微批次事件循环：`_pp_send_pyobj_to_next_stage()`、`_pp_launch_batch()`、`_pp_process_batch_result()`、`_pp_commit_comm_work()` |
| `python/sglang/srt/managers/scheduler.py` | 将 PP mixin 集成到主调度循环；分块预填充大小管理 |
| `python/sglang/srt/distributed/parallel_state.py` | PP 进程组初始化、`pp_size`、阶段 rank 分配 |
| `python/sglang/srt/server_args.py` | 定义 `--pp-size`、`--chunked-prefill-size`、`--enable-dynamic-chunking` 标志 |
| `python/sglang/srt/entrypoints/http_server.py` | PP 部署的多节点启动协调 |

### 架构

```
[Stage 0: GPU 0-7]     [Stage 1: GPU 8-15]    [Stage 2: GPU 16-23]   [Stage 3: GPU 24-31]
  Layers 0-14            Layers 15-29           Layers 30-44           Layers 45-60
       │                      │                      │                      │
       │── async P2P send ──▶ │── async P2P send ──▶ │── async P2P send ──▶ │
       │                      │                      │                      │
  forward_stream         forward_stream          forward_stream         forward_stream
  copy_stream            copy_stream             copy_stream            copy_stream
  default_stream(sync)   default_stream(sync)    default_stream(sync)   default_stream(sync)
```

### 关键代码逻辑

- **异步 P2P 通信**: `_pp_send_pyobj_to_next_stage()` 使用 `async_send` 返回 `P2PWork` 句柄；同步延迟到 `_pp_commit_comm_work()` 以实现 CPU/GPU 重叠
- **多流执行**: 专用的 `forward_stream`（GPU 计算）、`copy_stream`（D2H 传输）和 `default_stream`（同步）以实现最大重叠
- **动态分块**: 二次运行时模型基于前缀长度预测最优块大小，对齐到 `max(--page-size, 64)` 的倍数；由 `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`（默认 0.75）控制

### 集成要点

- **服务器标志**: `--pp-size N`、`--chunked-prefill-size`、`--enable-dynamic-chunking`、`--nnodes`、`--node-rank`、`--dist-init-addr`
- **环境变量**: `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR` (0.0-1.0)、`SGLANG_PP_LAYER_PARTITION`（例如 `15,15,15,16`）
- **与 PD 分离兼容**: PP 可以与 `--disaggregation-mode prefill/decode` 结合用于长上下文服务
