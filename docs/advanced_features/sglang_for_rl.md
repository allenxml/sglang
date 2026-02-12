# SGLang for RL Systems

This document is a practical guide for infrastructure teams integrating SGLang into RL and post-training systems. It focuses on the operational pain points in the loop (rollout, evaluation, training, weight sync) and maps them to concrete SGLang APIs, flags, and integration patterns. The focus is on maximizing rollout efficiency, accuracy and stability while keeping rollout-serving behavior aligned in production environments.

**中文对照**：# SGLang 用于强化学习系统

本文档是将 SGLang 集成到强化学习和后训练系统的 infrastructure 团队的实用指南。它关注循环中的操作痛点（ rollout、评估、训练、权重同步），并将它们映射到具体的 SGLang API、标志和集成模式。重点是最大化 rollout 效率、准确性和稳定性，同时保持生产环境中 rollout 服务行为的一致性。

## Why SGLang for RL Lifecycle?

Let's embrace a guiding principle from early DeepMind's RL engineering:

**Be a library, not a framework.**

This philosophy empowers innovation by providing SGLang as flexible tools, not rigid structures. Here are five reasons to use SGLang for your RL lifecycle:

* **Fine-Grained Engine Sleep and Wake Up**: facilitate maximum-powered rollout and training
* **Open-To-Use Refit Functionality**: diverse methods for co-location or disaggregation
* **Easy To Postpone Generation**: enable partial rollout and dedicated rollout control
* **Deterministic Inference**: achieve deterministic inference to enable zero training-inference mismatch
* **Load Balancing Router**: cache-aware load-balancing for high-throughput rollout

The following sections cover these aspects in detail.

## Fine-Grained Engine Sleep and Wake Up

Rollout and training are both memory-intensive, and co-locating them on the same GPUs often leads to memory pressure and slow handoffs. SGLang provides a memory-aware sleep/wake mechanism that releases KV cache and weights while keeping the server process alive, then resumes them for rollout without a full restart. This avoids repeated disk I/O and CUDA graph recapture during each RL step.

Under the hood, the RL team uses CUDA-graph-aware weight offload via [torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver) to preserve virtual memory addresses for graph replay. For details, see: [Efficient RL Training - Optimizing Memory Usage in verl](https://hebiao064.github.io/rl-memory-management).

### Server flag

Enable memory saver support when launching the server:

```
--enable-memory-saver
```

### Release Memory

**Endpoint:** `POST /release_memory_occupation`

**Request body:**

| Field | Description | Defaults | Options |
| --- | --- | --- | --- |
| `tags` | Which memory regions to release. If omitted, all are released. | `None` | Type: list[str], values: `kv_cache`, `weights` |
<!-- python/sglang/srt/managers/io_struct.py#L1381 currently only supports `kv_cache`, `weights` -->
**Behavior notes:**

- This call asserts there are no ongoing requests. Ensure the engine is idle before calling it.
- If `kv_cache` is released, SGLang flushes cache; subsequent requests will rebuild KV cache as needed.

### Resume Memory

**Endpoint:** `POST /resume_memory_occupation`

**Request body:**

| Field | Description | Defaults | Options |
| --- | --- | --- | --- |
| `tags` | Which memory regions to resume. If omitted, all are resumed. | `None` | Type: list[str], values: `kv_cache`, `weights` |
<!-- python/sglang/srt/managers/io_struct.py#L1393 currently only supports `kv_cache`, `weights` -->

## Open-To-Use Refit Functionality

After training completes each step, rollout engines must be refit with new weights. SGLang supports three refit strategies so you can match your infrastructure style (co-located vs disaggregated) and scaling needs. Each strategy maps to a concrete API with clear request schemas. For a deeper dive into SGLang's weight update utilities, see [RL System Deep Thinking: Weight Update Mechanisms](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/sys-design/readme-1-EN.md).

**How to choose:**

- **From disk** is simplest and best for elastic rollout scaling and checkpointing.
- **From tensor** is best for co-located training/rollout when you can pass in-memory tensors.
- **From distributed** is best for disaggregated training/rollout with dedicated communication groups (NCCL/IB).

### Update Weights from Disk

**When to use:**

- Save checkpoint to disk and update weights from disk
- Dynamic scaling (new rollout instances can load from the same checkpoint)

**Why it works well:**

This path trades some I/O overhead for simplicity and flexibility. It integrates naturally with checkpointing and makes it trivial to add new rollout engines: point them at the same checkpoint and call the API. It is also the safest option for high availability because the checkpoint itself is the source of truth.

**Endpoint:** `POST /update_weights_from_disk`

**Request body:**

| Field | Description | Defaults | Options |
| --- | --- | --- | --- |
| `model_path` | The model path with the new weights. | Required | Type: str |
| `load_format` | The format to load the weights. | `None` | Type: str |
| `abort_all_requests` | Abort all running requests before update. | `False` | Type: bool |
| `weight_version` | Optional weight version label tracked by the server. | `None` | Type: str |
| `is_async` | Perform weight load asynchronously. | `False` | Type: bool |
| `torch_empty_cache` | Empty torch cache. | `False` | Type: bool |
| `keep_pause` | Keep scheduler paused after update. | `False` | Type: bool |
| `recapture_cuda_graph` | Recapture CUDA graphs after update. | `False` | Type: bool |
| `token_step` | Trainer step id for rollout bookkeeping. | `0` | Type: int |
| `flush_cache` | Flush KV cache after update. | `True` | Type: bool |

**Response body:**

| Field | Description | Defaults | Options |
| --- | --- | --- | --- |
| `success` | Whether the update succeeded. | - | Type: bool |
| `message` | Status / error message. | - | Type: str |
| `num_paused_requests` | Number of paused requests during update. | `0` | Type: int |

**Python Engine API:** `engine.update_weights_from_disk(model_path, load_format=None)`

### Update Weights from Tensor

**When to use:**

- Co-located training and rollout, where training can provide tensors directly
- Fast in-memory updates

**Important constraints:**

This strategy requires the training process and rollout engine to share access to the tensors. Co-located setups must keep the model on GPU; moving tensors to CPU will break the update path. For high-performance MoE or specialized attention kernels, co-location may limit some optimizations compared to disaggregated rollouts.

**Endpoint:** `POST /update_weights_from_tensor`

**Request body:**

| Field | Description | Defaults | Options |
| --- | --- | --- | --- |
| `serialized_named_tensors` | Per-TP serialized tensor payloads. | Required | Type: list[str|bytes] |
| `load_format` | Optional load format selector. | `None` | `None`, `direct`, `flattened_bucket`, or a custom loader path string |
| `flush_cache` | Flush KV cache after update. | `True` | Type: bool |
| `abort_all_requests` | Abort all running requests before update. | `False` | Type: bool |
| `weight_version` | Optional version label tracked by the server. | `None` | Type: str |

**Note:** The serialized tensor payloads must be created with `MultiprocessingSerializer.serialize(...)` and should be base64-safe strings.

**Python Engine API:** `engine.update_weights_from_tensor(named_tensors, load_format=None, flush_cache=True)`

### Update Weights from Distributed Group

**When to use:**

- Disaggregated training and rollout
- NCCL or IB-backed weight broadcast from training workers to rollout workers

**How it works:**

Training workers gather weights (typically on TP rank 0), broadcast them to the rollout group, and each rollout TP shard loads the parameters it needs. This avoids disk I/O and keeps training and rollout decoupled, at the cost of managing a dedicated communication group.

**Initialize weight update group**

**Endpoint:** `POST /init_weights_update_group`

**Request body:**

| Field | Description | Defaults | Options |
| --- | --- | --- | --- |
| `master_address` | Group master address. | Required | Type: str |
| `master_port` | Group master port. | Required | Type: int |
| `rank_offset` | Offset for local rank mapping. | Required | Type: int |
| `world_size` | Total world size. | Required | Type: int |
| `group_name` | Group name. | `weight_update_group` | Type: str |
| `backend` | Communication backend. | `nccl` | Type: str |

**Update weight**

**Endpoint:** `POST /update_weights_from_distributed`

**Request body:**

| Field | Description | Defaults | Options |
| --- | --- | --- | --- |
| `names` | Parameter names to update. | Required | Type: list[str] |
| `dtypes` | Dtype strings for each parameter. | Required | Type: list[str] |
| `shapes` | Tensor shapes. | Required | Type: list[list[int]] |
| `group_name` | Group name. | `weight_update_group` | Type: str |
| `flush_cache` | Flush KV cache after update. | `True` | Type: bool |
| `abort_all_requests` | Abort all running requests before update. | `False` | Type: bool |
| `weight_version` | Optional version label. | `None` | Type: str |
| `load_format` | Optional format selector. | `None` | `None` or `flattened_bucket` |

**Destroy weights update group**

**Endpoint:** `POST /destroy_weights_update_group`

**Request body:**

| Field | Description | Defaults | Options |
| --- | --- | --- | --- |
| `group_name` | Group name. | `weight_update_group` | Type: str |

**Python Engine APIs:**

- `engine.init_weights_update_group(...)`
- `engine.update_weights_from_distributed(names, dtypes, shapes, ...)`
- `engine.destroy_weights_update_group(group_name)`

## Easy To Postpone Generation

Multi-turn RL rollouts often suffer from long-tail requests that block the entire batch. A small number of slow interactions can stall all GPUs, and the long-tail behavior makes profiling and monitoring difficult.

SGLang exposes explicit pause/resume APIs so you can pause slow requests and continue them later. This pattern matches systems like [APRIL](https://arxiv.org/abs/2509.18521), terminate once enough responses are collected, and recycle incomplete responses in the next step. The result is higher GPU utilization without discarding partial work.

`pause_generation` ---  update weights --- `continue_generation` is the correct execution flow when updating weights from training. An update can only happen when SGLang is not actively processing inference tasks.

### Pause Generation

**Endpoint:** `POST /pause_generation`

**Request body:**

| Field | Description | Defaults | Options |
| --- | --- | --- | --- |
| `mode` | Pause mode. | `abort` | `abort`, `retract`, `in_place` |

**Modes:**

- `abort`: Default behavior, identical to `abort` endpoint with `abort_all` set. Pending requests from `waiting_queue` and `running_queue` will be returned immediately to the caller.
- `retract`: Put engine in "paused" state.  Move running requests back to waiting queue. KV cache can be flushed and recomputed later.
- `in_place`: Put engine in "paused" state without changing states of the requests. Running requests rely on availability of KV caches to continue, so any subsequent `flush_cache` call will be unsuccessful.

### Continue Generation

**Endpoint:** `POST /continue_generation`

## Deterministic Inference

In many RL stacks, rollout and training are implemented with different kernels or batching behavior. Even when weights are identical, token probabilities can drift, silently breaking the on-policy assumption. This is the training–inference mismatch problem.

SGLang supports a deterministic inference mode that reduces non-determinism across batch shapes. This mitigates variance introduced by runtime batching and kernel selection. To further achieve true on-policy training, you need to modify the training engine to use the same deterministic kernels. For implementation details, see these miles examples: [True On-Policy](https://github.com/radixark/miles/tree/main/examples/true_on_policy) and [True On-Policy for VLM](https://github.com/radixark/miles/tree/main/examples/true_on_policy_vlm). For additional context, see the blog post [Let Speed Be With Stability: All-In-One Solution to Training-Inference Mismatch with Miles](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/mismatch/blog-en.md).

**Server flag:**

```
--enable-deterministic-inference
```

For more details, see [Deterministic Inference](deterministic_inference.md)

## Load Balancing Router

SGLang Model Gateway is the recommended control plane for large‑scale RL rollouts. It provides async, non‑blocking request handling, cache‑aware load balancing, and fault‑tolerant routing across rollout and reward servers. This lets you keep GPUs saturated while avoiding long‑tail stalls and brittle, engine‑local concurrency logic. It has been deployed in the training of GLM 4.5+ models and proven to be highly efficient in production-level large-scale RL workloads.

Key benefits for RL infrastructure:

- **Async non-blocking efficiency**: SGLang’s native async server/router architecture (HTTPS/gRPC) manages concurrency automatically. This guarantees maximum GPU saturation and effective continuous batching without requiring complex, manual implementation by engineers.
- **Elasticity and fault tolerance**: By encapsulating the reward model and rollout as independent servers, SGLang decouples them logically and physically. This architecture provides robust disaster recovery for large-scale distributed training; if a server fails, the router automatically redirects traffic to healthy nodes, ensuring the training process continues without interruption.
- **Training–Inference alignment**: Using the SGLang Model Gateway for both training and inference ensures "What You See Is What You Get." This eliminates score discrepancies and the painful backend alignment issues often caused by using different engines for training versus deployment.
- **Dynamic load balancing and long-tail mitigation**: Unlike static partitioning, the SGLang Model Gateway enables request-level dynamic dispatching for multi-turn RL. It can distribute different turns of a conversation across different servers to balance workloads and eliminate long-tail latency caused by varying sequence lengths.

For deployment and configuration, see: [SGLang Model Gateway](sgl_model_gateway.md)

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/entrypoints/http_server.py` | 暴露 `/release_memory_occupation`、`/resume_memory_occupation`、`/pause_generation`、`/continue_generation`、`/update_weights_from_disk`、`/update_weights_from_tensor`、`/update_weights_from_distributed` 端点 |
| `python/sglang/srt/entrypoints/engine.py` | Python Engine API: `engine.update_weights_from_disk()`、`engine.update_weights_from_tensor()`、`engine.init_weights_update_group()` 等 |
| `python/sglang/srt/managers/scheduler.py` | 处理睡眠/唤醒内存管理、暂停/恢复状态转换、空闲检查 |
| `python/sglang/srt/managers/scheduler_update_weights_mixin.py` | 权重更新逻辑：磁盘加载、张量反序列化、NCCL 分布式广播 |
| `python/sglang/srt/managers/io_struct.py` | 定义 `ReleaseMemoryOccupationReqInput`、`ResumeMemoryOccupationReqInput`、`PauseGenerationReqInput`、`UpdateWeightsFromDiskReqInput` 等 |
| `python/sglang/srt/managers/tokenizer_communicator_mixin.py` | 通过 `_Communicator` 将 HTTP 的 RL 控制请求路由到 Scheduler |

### 架构

```
[RL Training Loop]
       │
  ┌────┴────────────────────────────────────┐
  │  1. pause_generation (abort/retract/    │
  │     in_place)                           │
  │  2. release_memory_occupation           │
  │     (释放 KV cache + 权重以供            │
  │      训练使用 GPU 内存)                  │
  │  3. >>> 训练步骤 <<<                    │
  │  4. resume_memory_occupation            │
  │  5. update_weights_from_{disk|tensor|   │
  │     distributed}                        │
  │  6. continue_generation                 │
  │  7. >>> Rollout 步骤 <<<                │
  └─────────────────────────────────────────┘
```

### 关键代码逻辑

- **睡眠/唤醒**: `--enable-memory-saver` 通过 `torch_memory_saver` 启用 CUDA-graph 感知的权重卸载；`release_memory_occupation` 释放 KV cache 和/或权重，同时保持服务器进程存活
- **三种重载策略**: `update_weights_from_disk`（基于 checkpoint）、`update_weights_from_tensor`（内存中共位）、`update_weights_from_distributed`（用于解耦设置的 NCCL 广播）
- **暂停模式**: `abort`（返回待处理请求）、`retract`（将运行中的请求移至等待队列）、`in_place`（冻结而不改变状态）
- **确定性推理**: `--enable-deterministic-inference` 减少跨批次形状的非确定性，以实现训练-推理对齐

### 集成要点

- **服务器标志**: `--enable-memory-saver`、`--enable-deterministic-inference`
- **权重更新组**: `init_weights_update_group()` → `update_weights_from_distributed()` → `destroy_weights_update_group()` 生命周期用于基于 NCCL 的重载
- **RL 框架**: 通过 HTTP API 或 Python Engine API 与 verl、miles 和其他 RL 训练系统兼容
- **负载均衡**: SGLang Model Gateway 为多服务器 rollout 部署提供缓存感知路由
