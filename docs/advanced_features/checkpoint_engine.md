# Checkpoint Engine Integration

The SGLang checkpoint engine integration provides an efficient way to load model weights using a distributed checkpoint loading system. This feature significantly reduces model loading time, especially for large models and multi-node setups, by parallelizing the weight loading process across multiple processes and nodes.

**中文对照**：SGLang 检查点引擎集成提供了一种使用分布式检查点加载系统来加载模型权重的高效方法。此功能通过跨多个进程和节点并行化权重加载过程，显著减少了模型加载时间，特别是对于大型模型和多节点设置。

## Overview

The checkpoint engine integration allows SGLang to:
- Load model weights in parallel using multiple processes
- Distribute weight loading across multiple nodes to increase effective disk bandwidth
- Overlap weight loading with other initialization tasks like CUDA graph capture
- Support both single-node and multi-node deployments

**中文对照**：## 概述

检查点引擎集成允许 SGLang：
- 使用多个进程并行加载模型权重
- 跨多个节点分布权重加载以增加有效磁盘带宽
- 将权重加载与其他初始化任务（如 CUDA graph 捕获）重叠
- 支持单节点和多节点部署

## Installation

First, install the checkpoint engine package:

```bash
pip install 'checkpoint-engine[p2p]'
```

## Architecture

The system consists of two main components:

1. **SGLang Server**: Runs with `--wait-for-initial-weights` flag to wait for weights before becoming ready
2. **Checkpoint Engine Workers**: Separate processes (managed by torchrun) that load and distribute model weights

The checkpoint engine uses a parameter server architecture with support for:
- **Broadcast mode**: Weights are broadcast from loading processes to inference processes
- **P2P mode**: Direct peer-to-peer weight transfer between processes
- **All mode**: Combination of both broadcast and P2P methods

**中文对照**：该系统由两个主要组件组成：

1. **SGLang 服务器**：使用 `--wait-for-initial-weights` 标志运行，以在就绪前等待权重
2. **检查点引擎工作进程**：独立进程（由 torchrun 管理），用于加载和分发模型权重

检查点引擎使用参数服务器架构，支持：
- **广播模式**：权重从加载进程广播到推理进程
- **P2P 模式**：进程之间的直接点对点权重传输
- **全部模式**：广播和 P2P 方法的组合

## Usage Examples

### Single Node Setup

**Terminal 1 - Launch SGLang Server:**
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights
```

**Terminal 2 - Run Checkpoint Engine:**

Using sglang entrypoint:
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

Using torchrun directly:
```bash
torchrun --nproc-per-node 8 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

### Multi-Node Setup (2 Nodes)

**Node 0:**

Launch SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP]
```

Run checkpoint engine:

Using sglang entrypoint (recommended):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

Using torchrun directly:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

**Node 1:**

Launch SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP]
```

Run checkpoint engine:

Using sglang entrypoint (recommended):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

Using torchrun directly:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

### Multi-Node Setup with Tensor Parallelism (TP=16)

**Node 0:**

Launch SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP] \
    --dist-init-addr [IP]:9120 \
    --nnodes 2 \
    --node-rank 0
```

Run checkpoint engine:

Using sglang entrypoint (recommended):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

Using torchrun directly:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

**Node 1:**

Launch SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP] \
    --dist-init-addr [IP]:9120 \
    --nnodes 2 \
    --node-rank 1
```

Run checkpoint engine:

Using sglang entrypoint (recommended):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

Using torchrun directly:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

## Configuration Options

### SGLang Server Options

- `--load-format dummy`: Use dummy format for initial loading (allows overlapping with other tasks)
- `--wait-for-initial-weights`: Wait for checkpoint engine to provide weights before becoming ready
- `--host`: Host address for multi-node setups
- `--dist-init-addr`: Distributed initialization address for tensor parallelism

### Checkpoint Engine Options

- `--update-method`: Weight update method (`broadcast`, `p2p`, or `all`)
- `--checkpoint-path`: Path to model checkpoint directory
- `--inference-parallel-size`: Number of inference parallel processes
- `--endpoint`: SGLang server endpoint (default: `http://localhost:19730`)
- `--checkpoint-name`: Name for the checkpoint (default: `my-checkpoint-iter-0`)
- `--save-metas-file`: File to save checkpoint metadata
- `--load-metas-file`: File to load checkpoint metadata from
- `--uds`: Unix domain socket path for communication
- `--weight-version`: Version identifier for weights

## Performance Benefits

The checkpoint engine provides significant time savings in two main aspects:

1. **Multi-node Loading**: Each node only loads a portion of weights from disk, effectively increasing disk bandwidth. More participating nodes provide greater acceleration. Preliminary tests show 20-second acceleration when loading DeepSeek-R1 on H20-3e with two nodes.

2. **Single Process Optimization**: Using dummy format allows overlapping disk-to-CPU transfer with CUDA graph capture and other initialization tasks, providing additional time savings.

**中文对照**：检查点引擎在两个方面提供了显著的时间节省：

1. **多节点加载**：每个节点只从磁盘加载一部分权重，有效增加磁盘带宽。参与的节点越多，加速越大。初步测试显示，在 H20-3e 上使用两个节点加载 DeepSeek-R1 时可加速 20 秒。

2. **单进程优化**：使用虚拟格式允许将磁盘到 CPU 的传输与 CUDA graph 捕获和其他初始化任务重叠，提供额外的时间节省。

## Troubleshooting

- Ensure checkpoint engine package is installed: `pip install 'checkpoint-engine[p2p]'`
- Verify network connectivity between nodes in multi-node setups
- Check that the checkpoint path contains valid model files
- Monitor logs for connection errors between SGLang server and checkpoint engine
- Use `--sleep-time` parameter to add delays if needed for debugging

## 代码实现

### 核心文件

本功能主要由以下文件实现：

1. **[python/sglang/srt/checkpoint_engine/update.py](../../python/sglang/srt/checkpoint_engine/update.py)**
   - 主要职责：Checkpoint Engine 的启动入口和权重加载协调器
   - 关键类：`ParameterServer`（来自 checkpoint-engine 包）
   - 关键函数：`main()` - 启动权重更新流程，`check_sglang_ready()` - 检查服务器就绪状态

2. **[python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py](../../python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py)**
   - 主要职责：单个 worker 的权重加载和分发逻辑
   - 关键功能：从磁盘读取权重并通过 P2P/Broadcast 分发到推理进程

3. **[python/sglang/srt/managers/scheduler_update_weights_mixin.py](../../python/sglang/srt/managers/scheduler_update_weights_mixin.py)**
   - 主要职责：调度器端的权重更新接口（Mixin 模式）
   - 关键类：`SchedulerUpdateWeightsMixin`
   - 关键方法：
     - `update_weights_from_disk()` - 从磁盘更新权重
     - `init_weights_update_group()` - 初始化权重更新组
     - `update_weights_from_distributed()` - 从分布式源更新权重

4. **[python/sglang/srt/managers/io_struct.py](../../python/sglang/srt/managers/io_struct.py)**
   - 主要职责：定义权重更新相关的请求和响应数据结构
   - 关键类：`UpdateWeightFromDiskReqInput`、`InitWeightsUpdateGroupReqInput` 等

### 架构

权重加载流程的模块调用关系：

```
1. 启动 SGLang Server
   └─> engine.py (with --wait-for-initial-weights flag)
       └─> Scheduler (waiting for weights)

2. 启动 Checkpoint Engine
   └─> checkpoint_engine/update.py::main()
       ├─> ParameterServer.init() (setup distributed group)
       ├─> check_sglang_ready() (wait for server to be ready)
       └─> ParameterServer.broadcast_parameters()
           └─> Parallel loading from disk (multi-node/multi-process)

3. 权重传输
   └─> checkpoint_engine → SGLang Server
       └─> scheduler_update_weights_mixin.py
           └─> tp_worker.update_weights_from_disk()
               └─> model_runner.py::load_weights()

4. 服务器就绪
   └─> Scheduler.flush_cache() (if needed)
   └─> Server becomes ready for inference
```

### 关键代码逻辑

**功能点 1：服务器等待权重标志**
- 实现位置：`server_args.py` 定义参数，`engine.py` 检查标志
- 核心逻辑：`--wait-for-initial-weights` 标志让服务器在启动时不立即加载权重，而是等待 checkpoint engine 推送

**功能点 2：权重更新入口**
- 实现位置：`scheduler_update_weights_mixin.py:46-57`（`update_weights_from_disk()`）
- 核心逻辑：调用 `tp_worker.update_weights_from_disk()`，成功后可选地刷新缓存

**功能点 3：分布式权重加载**
- 实现位置：`checkpoint_engine/update.py:main()`
- 核心逻辑：使用 torchrun 启动多进程，每个进程加载部分权重（分片），然后通过 ParameterServer 分发到推理进程

**功能点 4：权重分发模式**
- 实现位置：`checkpoint_engine/update.py` 的 `--update-method` 参数
- 支持模式：
  - `broadcast`：权重从加载进程广播到推理进程
  - `p2p`：点对点传输（更高效）
  - `all`：混合模式

### 集成要点

**配置参数**：
- `server_args.py` 中的关键参数：
  - `--wait-for-initial-weights`：让服务器等待外部权重加载
  - `--load-format dummy`：使用占位符初始化模型，不从磁盘加载

**启动流程**：
- `engine.py`：检查 `--wait-for-initial-weights`，如果设置则跳过初始权重加载
- `checkpoint_engine/update.py`：作为独立进程启动，负责加载和分发权重

**运行时交互**：
- Checkpoint Engine 通过 HTTP API 与 SGLang Server 通信
- Endpoint：默认 `http://localhost:19730`，可通过 `--endpoint` 参数配置
- 权重传输通过 IPC（进程间通信）或 Distributed（分布式通信）完成

**性能优化**：
- **多节点加载**：每个节点只加载部分权重，有效增加磁盘带宽
- **重叠加载**：使用 `dummy` 格式时，可以重叠 CUDA graph 捕获和权重加载
- **零拷贝传输**：权重传输使用共享内存或 RDMA（取决于 checkpoint-engine 配置）

## References

- [Checkpoint Engine Repository](https://github.com/MoonshotAI/checkpoint-engine)
