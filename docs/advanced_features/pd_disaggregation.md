# PD Disaggregation

## Why and What is PD Disaggregation?

Large Language Model (LLM) inference comprises two distinct phases: **Prefill** and **Decode**. The Prefill phase is computation-intensive, processing the entire input sequence, while the Decode phase is memory-intensive, managing the Key-Value (KV) cache for token generation. Traditionally, these phases are handled within a unified engine, where combined scheduling of prefill and decode batches introduces inefficiencies. To address these challenges, we introduce **Prefill and Decoding (PD) Disaggregation** in SGLang.

### Issues with Unified Scheduling

The conventional unified engine, which processes prefill and decode batches together, results in two significant problems:

1. **Prefill Interruption**: Incoming prefill batches frequently interrupt ongoing decode batches, causing substantial delays in token generation.
2. **DP Attention Imbalance**: In data-parallel (DP) attention, one DP worker may process a prefill batch while another handles a decode batch simultaneously, leading to increased decode latency.

PD Disaggregation resolves these by separating the two stages, enabling tailored optimizations for each.

For the design details, please refer to [link](https://docs.google.com/document/d/1rQXJwKd5b9b1aOzLh98mnyMhBMhlxXA5ATZTHoQrwvc/edit?tab=t.0).

Currently, we support Mooncake and NIXL as the transfer engine.

## Profiling in PD Disaggregation Mode

When you need to profile prefill or decode workers in PD disaggregation mode, please refer to the [Profile In PD Disaggregation Mode](https://docs.sglang.io/developer_guide/benchmark_and_profiling.html#profile-in-pd-disaggregation-mode) section in the Benchmark and Profiling guide. Due to torch profiler limitations, prefill and decode workers must be profiled separately using dedicated command-line options.

## Router Integration

For deploying PD disaggregation at scale with load balancing and fault tolerance, SGLang provides a router. The router can distribute requests between prefill and decode instances using various routing policies. For detailed information on setting up routing with PD disaggregation, including configuration options and deployment patterns, see the [SGLang Model Gateway (former Router)](../advanced_features/sgl_model_gateway.md#prefill-decode-disaggregation).

**中文对照**：# PD 分离

## 为什么以及什么是 PD 分离？

大语言模型（LLM）推理包含两个截然不同的阶段：**预填充**和解码。预填充阶段是计算密集型的，处理整个输入序列，而解码阶段是内存密集型的，管理用于令牌生成的键值（KV）缓存。传统上，这些阶段在统一的引擎中处理，预填充和解码批次的组合调度会导致效率低下。为了应对这些挑战，我们在 SGLang 中引入了**预填充和解码（PD）分离**。

### 统一调度的问题

处理预填充和解码批次的传统统一引擎会导致两个重要问题：

1. **预填充中断**：传入的预填充批次经常中断正在进行的解码批次，导致令牌生成大量延迟。
2. **DP 注意力不平衡**：在数据并行（DP）注意力中，一个 DP 工作进程可能处理预填充批次，而另一个同时处理解码批次，导致解码延迟增加。

PD 分离通过分离两个阶段来解决这些问题，使每个阶段都能进行定制优化。

有关设计详情，请参阅[链接](https://docs.google.com/document/d/1rQXJwKd5b9b1aOzLh98mnyMhBMhlxXA5ATZTHoQrwvc/edit?tab=t.0)。

目前，我们支持 Mooncake 和 NIXL 作为传输引擎。

## PD 分离模式下的性能分析

当您需要对 PD 分离模式下的预填充或解码工作进程进行性能分析时，请参阅基准测试和性能分析指南中的[在 PD 分离模式下进行性能分析](https://docs.sglang.io/developer_guide/benchmark_and_profiling.html#profile-in-pd-disaggregation-mode)部分。由于 torch 性能分析器的限制，预填充和解码工作进程必须使用专用的命令行选项分别进行性能分析。

## 路由器集成

为了大规模部署具有负载均衡和容错能力的 PD 分离，SGLang 提供了一个路由器。路由器可以使用各种路由策略在预填充和解码实例之间分发请求。有关使用 PD 分离设置路由的详细信息，包括配置选项和部署模式，请参阅 [SGLang 模型网关（前路由器）](../advanced_features/sgl_model_gateway.md#prefill-decode-disaggregation)。


## Mooncake
### Requirements

```bash
uv pip install mooncake-transfer-engine
```

### Usage

### Llama Single Node

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

### DeepSeek Multi-Node

```bash
# prefill 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# prefill 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# decode 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
# decode 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
```
### Advanced Configuration

PD Disaggregation with Mooncake supports the following environment variables for fine-grained control over system behavior.

#### NVLink Transport Configuration
To enable NVLink transport for KV cache transfers with the mooncake backend (recommended for NVL72 deployments), set the following environment variables. Note that auxiliary data transfer will still use TCP as a temporary workaround.

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

The `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` environment variable enables the custom memory pool. Supported values are `NVLINK` (or `True`), `BAREX`, and `INTRA_NODE_NVLINK`.

#### Prefill Server Configuration
| Variable | Description | Default |
|:--------:|:-----------:|:--------:
| **`SGLANG_DISAGGREGATION_THREAD_POOL_SIZE`** | Controls the total number of worker threads for KVCache transfer operations per TP rank | A dynamic value calculated by `int(0.75 * os.cpu_count()) // 8)`, which is limited to be larger than 4 and less than 12 to ensure efficiency and prevent thread race conditions |
| **`SGLANG_DISAGGREGATION_QUEUE_SIZE`** | Sets the number of parallel transfer queues. KVCache transfer requests from multiple decode instances will be sharded into these queues so that they can share the threads and the transfer bandwidth at the same time. If it is set to `1`, then we transfer requests one by one according to fcfs strategy | `4` |
| **`SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`** | Timeout (seconds) for receiving destination KV indices during request initialization | `300` |

If a greater mean TTFT is acceptable, you can `export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600` (10 minutes) to relax the timeout condition.
Please be aware that this setting will cause prefill instances to take a longer time to clean up the affected memory resources when a running decode node loses connection.

#### Decode Server Configuration
| Variable | Description | Default |
|:--------:|:-----------:|:--------:
| **`SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL`** | Interval (seconds) between health checks to prefill bootstrap servers | `5.0` |
| **`SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE`** | Consecutive heartbeat failures before marking prefill server offline | `2` |
| **`SGLANG_DISAGGREGATION_WAITING_TIMEOUT`** | Timeout (seconds) for receiving KV Cache after request initialization | `300` |

If a greater mean TTFT is acceptable, you can `export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600` (10 minutes) to relax the timeout condition.


## NIXL
### Requirements

Install via pip.

```bash
pip install nixl
```

Or build from source - may be required if you already have UCX installed.

```bash
git clone https://github.com/ai-dynamo/nixl.git
cd nixl
pip install . --config-settings=setup-args="-Ducx_path=/path/to/ucx"
```


### Usage

### Llama Single Node

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-transfer-backend nixl
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-transfer-backend nixl
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

### DeepSeek Multi-Node

```bash
# prefill 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# prefill 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# decode 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
# decode 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
```

### Advanced Configuration

#### NIXL Backend Selection

By default, NIXL uses the **UCX** backend for KV cache transfers. You can select a different NIXL plugin backend depending on your infrastructure using the environment variable `SGLANG_DISAGGREGATION_NIXL_BACKEND`.

Example: `export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC`

**Available backends:** UCX (default), LIBFABRIC, or any installed NIXL plugin.

Example usage:
```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --port 30000
```

## ASCEND

### Usage

Use ascend backend with [memfabric_hybrid](https://gitcode.com/Ascend/memfabric_hybrid) and ASCEND_MF_STORE_URL being set

```bash
pip install memfabric-hybrid==1.0.0
export ASCEND_MF_STORE_URL="tcp://xxx.xx.xxx.xxx:xxxx"
```
Use mooncake backend, more details can be found in mooncake section.
```bash
export ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=true
```
ASCEND_NPU_PHY_ID need to be set in container env
```bash
export ASCEND_NPU_PHY_ID=xxx
```


### Llama Single Node

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-transfer-backend ascend
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-transfer-backend ascend
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

### DeepSeek Multi-Node

```bash
# prefill 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend ascend \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size 16
# decode 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend ascend \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size 16
```

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/disaggregation/prefill.py` | Prefill 端调度混入：`PrefillBootstrapQueue` 管理请求握手，`SchedulerDisaggregationPrefillMixin` 提供 prefill 专用事件循环 |
| `python/sglang/srt/disaggregation/decode.py` | Decode 端调度混入：`DecodePreallocQueue` 预分配内存，`DecodeTransferQueue` 轮询 KV 传输完成状态 |
| `python/sglang/srt/disaggregation/common/conn.py` | 通用连接层：`CommonKVManager`、`CommonKVSender`、`CommonKVReceiver`、`CommonKVBootstrapServer`，基于 ZMQ 的跨实例协调 |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | Mooncake 传输引擎：`MooncakeKVManager`、`MooncakeKVSender`、`MooncakeKVReceiver`，支持 GPU 直传和 NVLink 自定义内存池 |
| `python/sglang/srt/disaggregation/nixl/conn.py` | NIXL 传输引擎：`NixlKVManager`、`NixlKVSender`、`NixlKVReceiver`，支持 UCX 和 LIBFABRIC 后端 |
| `python/sglang/srt/disaggregation/ascend/conn.py` | Ascend NPU 传输引擎：使用 `memfabric_hybrid` 实现设备间 RDMA 传输 |
| `python/sglang/srt/disaggregation/base/conn.py` | 抽象基类：定义 `KVManager`、`KVSender`、`KVReceiver` 接口规范 |
| `python/sglang/srt/disaggregation/utils.py` | 工具类：`DisaggregationMode` 枚举、`MetadataBuffers` 元数据缓冲区、`ReqToMetadataIdxAllocator` 内存池索引分配 |
| `python/sglang/srt/managers/scheduler.py` | 调度器主体：通过混入模式集成 prefill/decode 专用事件循环 |
| `python/sglang/srt/server_args.py` | `--disaggregation-mode`、`--disaggregation-transfer-backend`、`--disaggregation-bootstrap-port` 等命令行参数定义 |

### 架构

```
                        ┌─────────────────────────────┐
                        │        Router（路由器）        │
                        │   sglang_router.launch_router │
                        └──────────┬──────────────────┘
                                   │ 请求分发
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
         ┌──────────────────┐           ┌──────────────────┐
         │  Prefill 实例     │           │  Decode 实例      │
         │  --disaggregation │           │  --disaggregation │
         │  -mode prefill    │           │  -mode decode     │
         ├──────────────────┤           ├──────────────────┤
         │ PrefillBootstrap  │◄─────────►│ DecodePrealloc   │
         │ Queue（握手队列） │  Bootstrap │ Queue（预分配）   │
         ├──────────────────┤  协调      ├──────────────────┤
         │ GPU Prefill 计算  │           │ DecodeTransfer   │
         │ → KV Cache 生成   │──────────►│ Queue（传输接收） │
         └──────────────────┘  Mooncake/ ├──────────────────┤
                                NIXL/    │ GPU Decode 计算   │
                                Ascend   │ → Token 生成      │
                                KV传输   └──────────────────┘
```

### 集成要点

- **可插拔传输后端**：通过 `--disaggregation-transfer-backend` 选择 mooncake（默认）、nixl、ascend 或 fake（测试用），所有后端实现统一的 `KVManager`/`KVSender`/`KVReceiver` 接口
- **调度器混入模式**：`SchedulerDisaggregationPrefillMixin` 和 `SchedulerDisaggregationDecodeMixin` 以混入方式注入 `Scheduler` 类，提供 prefill/decode 专用的 `event_loop_normal_disagg_*()` 和 `event_loop_overlap_disagg_*()` 事件循环
- **Bootstrap 握手**：每个请求通过唯一的 `bootstrap_room` 进行 Prefill↔Decode 元数据交换（目标 KV 索引、输出 token、logprobs），协调端口由 `--disaggregation-bootstrap-port`（默认 8998）控制
- **队列管理**：Prefill 端使用 `PrefillBootstrapQueue` → `disagg_prefill_inflight_queue` 两级队列；Decode 端使用 `DecodePreallocQueue` → `DecodeTransferQueue` → `waiting_queue` 三级队列
- **TP/DP 支持**：通过 `--disaggregation-decode-tp` 和 `--disaggregation-decode-dp` 允许 Prefill 和 Decode 实例使用不同的并行度配置
