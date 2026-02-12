# DP, DPA and SGLang DP Router

This guide explains the difference between Data Parallelism (DP) and Data Parallelism Attention (DPA), how to enable each mode correctly, and how to use the SGLang Model Gateway (SMG) for production-grade DP deployments.

**中文对照**：本指南解释了数据并行（DP）和数据并行注意力（DPA）之间的区别，如何正确启用每种模式，以及如何使用 SGLang 模型网关（SMG）进行生产级的 DP 部署。

## Data Parallelism (DP)

**Data Parallelism (DP)** is the most common parallelism strategy that replicates the entire model across multiple GPU sets and processes different batches of requests in parallel. Each GPU set handles independent requests. With dedicated routing strategies, as we will introduce later, with those proper routing algorithms in SGLang Model Gateway, the throughput of your serving system could be multiplied nearly linearly.

### Key characteristics

- Each replica has a full copy of the model
- Requests are distributed/scattered across replicas
- No inter-replica communication during one request's inference (for simple DP)

**中文对照**：## 数据并行（DP）

**数据并行（DP）** 是最常见的并行策略，它在整个模型复制到多组 GPU 上，并行处理不同的请求批次。每组 GPU 处理独立的请求。通过专用的路由策略，正如我们将在后面介绍的，通过 SGLang 模型网关中适当的路由算法，您的服务系统的吞吐量可以几乎线性地倍增。

### 关键特征

- 每个副本都有模型的完整副本
- 请求在副本之间分布/分散
- 在单个请求的推理过程中，副本之间没有通信（对于简单 DP）

## Data Parallelism Attention (DPA)

**Data Parallelism Attention (DPA)**, also known as DP Attention, is an advanced parallelism strategy. While DPA provides the most significant benefits for **Multi-Head Latent Attention (MLA)** models (such as DeepSeek, MiniMax, Kimi-K2), it also supports **standard attention models** like Qwen.

### The Problem with Tensor Parallelism for MLA Models

The most common parallelism strategy for inference is **Tensor Parallelism (TP)**. However, TP might not be the most efficient strategy for certain models. For example, DeepSeek models use MLA and only have **one KV head**. If we use tensor parallelism on 8 GPUs, it will lead to:

- **Duplicated KV cache** across all GPUs
- **Unwanted memory usage** that limits batch size
- **Reduced throughput** due to memory constraints

**中文对照**：## 数据并行注意力（DPA）

**数据并行注意力（DPA）**，也称为 DP 注意力，是一种高级并行策略。虽然 DPA 对**多头潜在注意力（MLA）**模型（如 DeepSeek、Minimax、Kimi-K2）提供最显著的益处，但它也支持像 Qwen 这样的**标准注意力模型**。

### MLA 模型的张量并行问题

推理最常见的并行策略是**张量并行（TP）**。然而，TP 可能不是某些模型的最有效策略。例如，DeepSeek 模型使用 MLA 并且只有**一个 KV 头**。如果我们在 8 个 GPU 上使用张量并行，将导致：

- **所有 GPU 上重复的 KV cache**
- **不必要的内存使用**限制了批量大小
- **由于内存限制导致吞吐量降低**

### How DPA Works

DPA addresses these limitations by applying **data parallelism specifically to the attention component**.

**Each DP replica:**

- Processes different batches independently (can be in different forward modes: prefill, decode, or idle)
- Maintains its own KV cache (no duplication)
- Enables significantly larger batch sizes due to memory savings

**Communication patterns in DPA + EP:**
-
-  **All2All (Dispatch)**: Routes tokens to expert sub-groups based on gating decisions
- **All2All (Combine)**: Gathers computed results from experts back to original token positions

### Key benefits of DPA

1. **Significantly reduced KV cache memory**: Each DP replica only stores KV cache for its own batches
2. **Larger batch sizes**: Memory savings enable larger batch sizes
3. **Improved decoding throughput**: Significant throughput gains for MLA-based models
4. **Independent forward modes**: Each DP replica can be in different forward modes (prefill, decode, or idle) and handles its assigned batches independently during attention computation

**中文对照**：### DPA 的工作原理

DPA 通过**专门对注意力组件应用数据并行**来解决这些限制。

**每个 DP 副本：**

- 独立处理不同的批次（可以处于不同的前向模式：预填充、解码或空闲）
- 维护自己的 KV cache（无重复）
- 由于内存节省，启用更大的批量大小

**DPA + EP 中的通信模式：**
-
- **All2All（分发）**：根据门控决策将令牌路由到专家子组
- **All2All（合并）**：收集专家的计算结果回到原始令牌位置

### DPA 的主要益处

1. **显著减少 KV cache 内存**：每个 DP 副本只为自己的批次存储 KV cache
2. **更大的批量大小**：内存节省启用更大的批量大小
3. **改进的解码吞吐量**：对基于 MLA 的模型有显著的吞吐量提升
4. **独立的前向模式**：每个 DP 副本可以处于不同的前向模式（预填充、解码或空闲），并在注意力计算期间独立处理其分配的批次

### DPA with Expert Parallelism for MoE

For MoE models like DeepSeek, DPA is **often** paired with Expert Parallelism (EP) for best throughput at scale. However, **DPA does not require EP**: you can enable DPA without EP if your deployment does not need expert sharding.

- Distribute 256+ expert weights across GPUs (cannot fit on a single GPU)
- Enable efficient all-to-all token routing via DeepEP
- Scale to large clusters (up to 5x throughput improvement over vanilla TP)

**中文对照**：### MoE 的 DPA 与专家并行

对于像 DeepSeek 这样的 MoE 模型，DPA **通常**与专家并行（EP）配对以获得最佳的规模化吞吐量。但是，**DPA 不需要 EP**：如果您的部署不需要专家分片，您可以启用 DPA 而不使用 EP。

- 跨 GPU 分布 256+ 专家权重（无法容纳在单个 GPU 上）
- 通过 DeepEP 启用高效的 all-to-all 令牌路由
- 扩展到大型集群（相比普通 TP 最高可达 5 倍吞吐量提升）

### Recommended setup for DeepSeek

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --dp-size 8 \
    --ep 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm
```

> **Note**: `--dp-size` must be explicitly set when using `--enable-dp-attention`. If `dp_size` is 1 (default), DPA will be disabled.

**中文对照**：> **注意**：使用 `--enable-dp-attention` 时必须显式设置 `--dp-size`。如果 `dp_size` 为 1（默认值），DPA 将被禁用。

For detailed EP configuration (DeepEP, Two-Batch Overlap, EPLB), see [Expert Parallelism](expert_parallelism.md).

### Target Models

DPA supports the following model architectures:

- **MLA (Multi-Head Latent Attention) models** - where DPA provides the most significant benefits:
  - DeepSeek family (DeepSeek-V2, DeepSeek-V3, DeepSeek-R1)
  - MiniMax models
  - Kimi-K2
  - Other models using MLA architecture

- **Standard attention models** - also supported:
  - Qwen models (see [PR #6121](https://github.com/sgl-project/sglang/pull/6121))

For models like Llama, with standard GQA, standard DP, or TP is typically recommended.

**中文对照**：DPA 支持以下模型架构：

- **MLA（多头潜在注意力）模型** - DPA 在这些模型上提供最显著的益处：
  - DeepSeek 系列（DeepSeek-V2、DeepSeek-V3、DeepSeek-R1）
  - MiniMax 模型
  - Kimi-K2
  - 其他使用 MLA 架构的模型

- **标准注意力模型** - 也支持：
  - Qwen 模型（参见 [PR #6121](https://github.com/sgl-project/sglang/pull/6121)）

对于像 Llama 这样具有标准 GQA 的模型，通常推荐使用标准 DP 或 TP。

To enable DPA, add `--enable-dp-attention` to your server launch command.

### Activation Logic

DPA is enabled explicitly via server arguments (CLI or config). You must set both `--dp-size` and `--enable-dp-attention`:

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --dp-size 8 \
    --enable-dp-attention
```

**Important**: `--dp-size` must be greater than 1 for DPA to work. When `dp_size == 1` (default), `--enable-dp-attention` is automatically disabled. The constraint `tp_size % dp_size == 0` must also be satisfied.

**中文对照**：DPA 通过服务器参数（CLI 或配置）显式启用。您必须同时设置 `--dp-size` 和 `--enable-dp-attention`：

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --dp-size 8 \
    --enable-dp-attention
```

**重要提示**：DPA 要工作，`--dp-size` 必须大于 1。当 `dp_size == 1`（默认值）时，`--enable-dp-attention` 会自动禁用。还必须满足约束条件 `tp_size % dp_size == 0`。

### Standard DP for MLA models

Note that MLA models, of course, also support DP. Suppose you want to enable standard DP for MLA models. First, launch each MLA model's replica independently. You may launch these replicas one by one with DPA enabled. After launching each MLA model's replica, launch an SMG and connect all the replicas to the SMG. A detailed explanation of SMG is as follows.

## Modern Data Parallelism SGLang Model Gateway (SMG)

### Native DP Mode

Native DP (built-in Data Parallelism) in SGLang creates multiple worker processes within a single SGLang instance, under the control of `DataParallelController` with the launching parameter of `dp-size`.


```bash
# Native DP mode
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4
```

**Limitations:**

- Built-in in-process load balancing only (e.g., `round_robin`, `total_requests`, `total_tokens`)
- No cache-aware routing
- Limited observability and metrics
- No fault tolerance or circuit breakers
- Not suitable for production workloads

⚠️ Native DP is **highly not recommended for use right now**. It is only used in some ancient/outdated RL frameworks. You can use SGLang Model Gateway (SMG) to power up your data parallelism in any use case.

### SMG-Based DP (Recommended)

Starting from September 2024, SGLang Model Gateway, i.e., SMG, formerly named as SGLang DP Router, was built especially as a production-ready DP routing system with Rust. It starts from DP routing, but later we further expanded its scope to coordinate RL, PD Disaggregation, and other scenarios. This doc only discusses SMG's usage in DP routing. For other usage, please refer to [SGLang Model Gateway Documentation](sgl_model_gateway.md).

> To achieve the best production-level routing performance and reduce the overhead to an extreme extent, we use Rust to build SMG, but not Python, since Python is never FAST enough.

**We strongly recommend using the SGLang Model Gateway (SMG) for production-grade Data Parallelism.** SMG provides significant advantages over native DP mode.

```bash
# SMG-based DP mode (Recommended)
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4
```

⚠️ Note that **SMG and Naive DP share the same launching parameter, `--dp-size`**. But the entrypoint of Naive DP is `python -m sglang.launch_server`, and SMG's entrypoint is `python -m sglang_router.launch_server`.

**Advantages of SMG-Based DP:**

| Feature | Native DP | SMG-Based DP |
|---------|-----------|--------------|
| **Load Balancing** | Built-in in-process methods | Advanced policies (cache-aware, power-of-two, etc.) |
| **Cache Awareness** | ❌ No | ✅ Yes - significantly higher cache hit rate |
| **Throughput** | Baseline | Significant improvement |
| **Multi-Node Support** | Limited | ✅ Full support |
| **Worker Health Monitoring** | Basic | ✅ Circuit breakers, health checks |
| **Reliability** | Basic | ✅ Retries, rate limiting, queuing |
| **Observability** | Basic metrics | ✅ 40+ Prometheus metrics, OpenTelemetry |
| **Hot Worker Add/Remove** | ❌ No | ✅ Yes |

###  SMG's Performance

The cache-aware routing policy in SMG significantly improves performance for workloads with shared prefixes:

| Metric | Without Cache-Aware | With Cache-Aware SMG |
|--------|---------------------|----------------------|
| Throughput (token/s) | 82,665 | 158,596 (+92%) |
| Cache Hit Rate | 20% | 75% (+275%) |

*Benchmark from [SGLang v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), workload with multiple long prefix groups, 8x A100 80GB GPUs, dp-size=8*

### When to Use Each

**Use Native DP when:**

- ~Never use Native/Naive DP~
- Learning material of DP routing

**Use SMG-Based DP when:**

- In any case, when you think DP is needed
- Production deployments
- Multi-node distributed setups
- Workloads with shared prefixes (high cache reuse potential)
- You need high availability and reliability features
- You require detailed observability and metrics

### Quick Start For SMG

**Installation**

```bash
pip install sglang-router
# or
pip install "sglang[all]"
```

**Option A: Co-launch Workers and SMG (Simplest)**

This is the easiest way to get started - SMG and workers are launched together:

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000
```

**Option B: Separate Launch (Multi-Node)**

For distributed deployments across multiple machines:

1. Launch workers on each node

```bash
# Node 1
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000

# Node 2
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000
```

2. Launch SMG pointing to workers

```bash
python -m sglang_router.launch_router \
    --worker-urls http://node1:8000 http://node2:8000 \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 30000
```

**Option C: Dynamic Worker Registration**

For elastic deployments where workers can be added/removed dynamically:

```bash
# Launch SMG first
python -m sglang_router.launch_router \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 30000

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
    -H "Content-Type: application/json" \
    -d '{"url": "http://worker1:8000"}'

curl -X POST http://localhost:30000/workers \
    -H "Content-Type: application/json" \
    -d '{"url": "http://worker2:8000"}'
```

### Load Balancing Policies

SMG supports multiple load balancing policies:

| Policy | Description | Best For |
|--------|-------------|----------|
| `cache_aware` | Combines cache locality with load balancing | **Recommended for most workloads** |
| `round_robin` | Cycles through workers in order | Simple, predictable distribution |
| `random` | Random worker selection | Baseline, testing |
| `power_of_two` | Samples two workers, picks lighter one | Low latency requirements |

**Cache-Aware Policy (Default, Recommended)**

The cache-aware policy provides the best performance for most workloads:

```bash
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware \
    --cache-threshold 0.5 \
    --balance-abs-threshold 32 \
    --balance-rel-threshold 1.5 \
    --eviction-interval-secs 120 \
    --max-tree-size 67108864
```

**How it works:**

1. Maintains an approximate radix tree for each worker based on request history
2. Routes requests to workers with the highest prefix match (cache hit)
3. Falls back to shortest-queue routing when load is imbalanced
4. Automatically evicts old entries to prevent memory overflow

### Best Practices

1. **Start with `cache_aware` policy** - It provides the best balance between cache locality and load distribution for most workloads
2. **Use SMG for production** - Prefer `sglang_router.launch_server` over `sglang.launch_server` for better reliability and observability
3. **Enable health checks** - Configure `--router-health-check-interval-secs` to detect and remove unhealthy workers automatically

**Recommended command with best practices applied:**

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --router-policy cache_aware \
    --router-health-check-interval-secs 30 \
    --router-prometheus-port 10001 \
    --host 0.0.0.0 \
    --port 30000
```

For advanced configuration (circuit breakers, retries, Prometheus metrics, K8s integration), see [SGLang Model Gateway Documentation](sgl_model_gateway.md).

### Verifying Traffic Distribution

After launching SMG, verify that traffic is being distributed correctly:

**1. Check worker status:**

```bash
curl http://localhost:30000/workers
```

**2. Check load distribution:**

```bash
curl http://localhost:30000/get_loads
```

**3. Monitor metrics (if Prometheus enabled):**

```bash
# Key metrics to check
smg_router_requests_total{model="..."}
smg_worker_requests_active{worker="..."}
sglang_cache_hit_rate{source="..."}
```

For detailed metrics and monitoring setup, see [SGLang Model Gateway Documentation](sgl_model_gateway.md).

## 代码实现

### 核心文件
- [../../python/sglang/srt/managers/data_parallel_controller.py](../../python/sglang/srt/managers/data_parallel_controller.py): Implements the native DP orchestrator that dispatches requests to multiple worker processes. It manages inter-process communication via ZMQ and applies load-balancing strategies to distribute traffic.
- [../../python/sglang/srt/managers/scheduler_dp_attn_mixin.py](../../python/sglang/srt/managers/scheduler_dp_attn_mixin.py): Provides the logic for DP Attention (DPA) synchronization. It defines the `MLPSyncBatchInfo` structure and handles collective communication (All-Gather) to align batch metadata across DP ranks.
- [../../python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py): Integrates DP and DPA logic by inheriting from the respective mixins. It coordinates the request lifecycle and triggers batch synchronization before model execution.
- **SGLang Model Gateway (SMG)**: Implemented in Rust (under `sgl-model-gateway/`), it provides a high-performance router with advanced cache-aware policies, intended for production-grade deployments.

### 架构
DP 架构使用集中式的 `DataParallelController`，它接收分词后的请求并将其路由到独立的 worker 副本。在标准 DP 中，workers 是隔离的。然而，在 **DPA（DP Attention）** 模式下，同一 TP 组内的 workers 在前向传播期间协作。这通过使用 NCCL 或基于 CPU 的 All-Gather 在所有 DP ranks 间同步批次元数据（token 数量、forward 模式）来实现。这种协调允许 MLA 模型在每个副本上存储唯一的 KV 缓存，同时共享权重，有效消除内存重复。

### 关键代码逻辑
DP 预算管理器根据最少的待处理 tokens 或请求数选择目标 worker：
```python
# From data_parallel_controller.py
target_rank = min(range(self.dp_size), key=lambda i: (self.total_tokens[i], self.total_requests[i]))
```
DPA 通过在进程组间收集批次信息确保所有副本对齐：
```python
# From scheduler_dp_attn_mixin.py
torch.distributed.all_gather_into_tensor(global_info_tensor.flatten(), local_info_tensor, group=group)
```
集成 SMG 允许跨多个节点进行可扩展的缓存感知路由：
```bash
# Integration via SMG launcher
python -m sglang_router.launch_server --model-path <path> --dp-size 4 --router-policy cache_aware
```

### 集成要点
- **--dp-size**：指定数据并行副本的数量
- **--enable-dp-attention**：在调度器和模型执行器中启用 DPA 同步逻辑
- **--load-balance-method**：配置原生 DP 路由策略（例如 `total_tokens`、`round_robin`）
- **SMG 配置**：使用 `sglang_router` 入口点获得生产级功能，如熔断器和 Prometheus 监控

## Reference

| Strategy | Use Case | Key Benefit |
|----------|----------|-------------|
| **Native DP** (`--dp-size`) | Never | Easy to understand, not rust based |
| **SMG-Based DP** | **Production (recommended)** | Cache-aware routing, high availability |
| **DPA** (`--dp-size N --enable-dp-attention`) | DeepSeek/MLA models | Eliminates KV cache duplication, improved throughput |
| **DPA + EP** | DeepSeek MoE models | Significant throughput improvement vs vanilla TP |

**Recommended production setup for DeepSeek:**
1. Enable **DPA** for attention layers (`--dp-size 8 --enable-dp-attention`)
2. Enable **EP** for MoE layers (`--ep 8 --moe-a2a-backend deepep`)
3. Use **SMG** with **cache_aware** policy

**Related documentation:**
- [Expert Parallelism](expert_parallelism.md) - DeepEP, Two-Batch Overlap, EPLB
- [SGLang Model Gateway Documentation](sgl_model_gateway.md) - SMG configuration & troubleshooting
- [Large-Scale EP Blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/) - 96 GPU deployment guide
