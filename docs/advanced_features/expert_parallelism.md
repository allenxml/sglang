# Expert Parallelism

Expert Parallelism (EP) in SGLang distributes expert weights across multiple devices in Mixture-of-Experts (MoE) models, addressing memory bottlenecks and enabling efficient scaling for high-performance inference. It is particularly vital for serving large-scale MoE models where tokens are dynamically routed to specialized experts across GPUs. By leveraging optimized all-to-all communication and grouped matrix multiplications (GEMMs), EP reduces latency, boosts throughput, and minimizes idle GPU time. SGLang's EP offers strong extensibility through its modular framework, allowing seamless integration of custom kernels, backends, and optimizations without refactoring core logic, supporting diverse hardware and quantization schemes.

**中文对照**：# 专家并行

SGLang 中的专家并行（EP）在混合专家（MoE）模型中将专家权重分布到多个设备上，解决内存瓶颈并实现高效的高性能推理扩展。对于服务大型 MoE 模型尤为重要，因为令牌会被动态路由到跨 GPU 的专门专家。通过利用优化的全对全通信和分组矩阵乘法（GEMM），EP 减少延迟、提高吞吐量并最小化 GPU 空闲时间。SGLang 的 EP 通过其模块化框架提供强大的可扩展性，允许无缝集成自定义内核、后端和优化，而无需重构核心逻辑，支持多种硬件和量化方案。

## Supported Backends and Selection Guidance

SGLang's EP integrates diverse, highly efficient backends for different use cases, allowing fine-grained control over performance trade-offs. Users specify backends via command-line flags:
- `--moe-a2a-backend`: Selects the backend for all-to-all communication.
- `--moe-runner-backend`: Selects the backend for MoE computation.

### Backends for All-to-All Communication

| Backend      | Description                                                                 | Use Cases                          |
|--------------|-----------------------------------------------------------------------------|------------------------------------|
| **`none` (default)** | Disables all-to-all for EP. Uses All-Reduce or All-Gather for token dispatch. | Hybrid EP and TP setups.           |
| `deepep`     | DeepEP, a communication library for efficient token shuffling in MoE models. | Large-scale EP deployments.        |
| `mooncake`   | An extension of DeepEP for elastic inference, leveraging RDMA for high-performance data transfers. | Elastic EP serving. |
| `mori` | MORI-EP, AMD's native all-to-all communication implementation optimized for ROCm. | AMD GPU deployments. |
| `flashinfer` | Flashinfer implementation of all-to-all. | Large-scale EP deployments. |
| `ascend_fuseep` | Ascend NPU native fused all-to-all communication. | Ascend NPU deployments. |

**中文对照**：## 支持的后端和选择指南

SGLang 的 EP 为不同的用例集成了多种高效后端，允许对性能权衡进行细粒度控制。用户通过命令行标志指定后端：
- `--moe-a2a-backend`：选择全对全通信的后端
- `--moe-runner-backend`：选择 MoE 计算的后端

### 全对全通信的后端

| 后端 | 描述 | 用例 |
|------|------|------|
| **`none`（默认）** | 为 EP 禁用全对全。使用 All-Reduce 或 All-Gather 进行令牌分发。 | 混合 EP 和 TP 设置。 |
| `deepep` | DeepEP，用于 MoE 模型中高效令牌重排的通信库。 | 大规模 EP 部署。 |
| `mooncake` | DeepEP 的扩展，用于弹性推理，利用 RDMA 进行高性能数据传输。 | 弹性 EP 服务。 |
| `mori` | MORI-EP，AMD 原生的全对全通信实现，针对 ROCm 进行了优化。 | AMD GPU 部署。 |
| `flashinfer` | Flashinfer 的全对全实现。 | 大规模 EP 部署。 |
| `ascend_fuseep` | Ascend NPU 原生融合全对全通信。 | Ascend NPU 部署。 |

DeepEP and Mooncake backends support two modes for token dispatch: `normal` mode (optimized for prefill workloads with high throughput) and `low_latency` mode (optimized for decode workloads with low latency and CUDA Graph compatibility). MORI backend only supports `normal` mode now. Users are recommended to set `--deepep-mode auto` to enable automatic dispatch mode switching during runtime. Setting `--deepep-mode normal` or `--deepep-mode low_latency` is useful for debugging or development purposes.

Currently, DeepEP, Mooncake, `ascend_fuseep` and MORI only support cases where `ep_size = tp_size`. For hybrid EP and TP (i.e., `ep_size < tp_size`), only the `none` backend (All-Reduce or All-Gather-based dispatching) is supported.

### Backends for MoE Computation

| Backend                  | Description                                                                 | Use Cases                          |
|--------------------------|-----------------------------------------------------------------------------|------------------------------------|
| **`auto` (default)**     | Automatically selects the optimal backend based on model architecture, hardware (e.g., NVIDIA architecture like Ampere, Hopper, Blackwell), quantization scheme (e.g., FP8, FP4), and runtime conditions. | General-purpose deployments; ensures compatibility and performance without user intervention. |
| `triton`                 | Triton-based implementation for grouped GEMMs. To achieve higher performance, it's highly recommended to create [tuned configurations](https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/fused_moe_triton/README.md). | Custom kernel development or scenarios requiring high extensibility with Torch compilation support. |
| `deep_gemm`              | DeepGEMM backend optimized for MoE matrix multiplications, supporting contiguous layouts for prefill and masked layouts for decode; often JIT-compiled for performance. | Large-scale EP deployments with FP8 block-wise quantization. |
| `cutlass`                | CUTLASS-based backend for efficient GEMMs. | NVIDIA architectures with CUTLASS support. |
| `flashinfer_trtllm`      | FlashInfer integrated with TensorRT-LLM for accelerated MoE computations, supporting FP4 communication operators and high-performance GEMMs. | Blackwell with TRT-LLM. |
| `flashinfer_cutlass`     | FlashInfer combined with CUTLASS for high-performance grouped GEMMs in MoE layers, handling FP4/FP8 quantization efficiently. | Blackwell with FP4/FP8 models. |
| `flashinfer_mxfp4`       | FlashInfer variant optimized for MXFP4 (mixed FP4) quantization in MoE runners, focusing on memory-efficient low-precision inference. | Low-precision models with MXFP4. |
| `flashinfer_cutedsl`     | FlashInfer with a custom DSL for flexible and efficient MoE kernel generation, integrated with ModelOpt FP4 quantization. | Low-precision models with NVFP4. |

**中文对照**：DeepEP 和 Mooncake 后端支持两种令牌分发模式：`normal` 模式（针对高吞吐量预填充工作负载进行优化）和 `low_latency` 模式（针对低延迟解码工作负载进行优化，支持 CUDA Graph 兼容性）。MORI 后端目前仅支持 `normal` 模式。建议用户设置 `--deepep-mode auto` 以在运行时启用自动分发模式切换。设置 `--deepep-mode normal` 或 `--deepep-mode low_latency` 对于调试或开发目的很有用。

目前，DeepEP、Mooncake、`ascend_fuseep` 和 MORI 仅支持 `ep_size = tp_size` 的情况。对于混合 EP 和 TP（即 `ep_size < tp_size`），仅支持 `none` 后端（基于 All-Reduce 或 All-Gather 的分发）。

### MoE 计算的后端

| 后端 | 描述 | 用例 |
|------|------|------|
| **`auto`（默认）** | 根据模型架构、硬件（如 NVIDIA Ampere、Hopper、Blackwell 架构）、量化方案（如 FP8、FP4）和运行时条件自动选择最优后端。 | 通用部署；确保兼容性和性能，无需用户干预。 |
| `triton` | 基于 Triton 的分组 GEMM 实现。要获得更高性能，强烈建议创建[调优配置](https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/fused_moe_triton/README.md)。 | 自定义内核开发或需要高可扩展性且支持 Torch 编译的场景。 |
| `deep_gemm` | DeepGEMM 后端，针对 MoE 矩阵乘法进行了优化，支持预填充的连续布局和解码的掩码布局；通常 JIT 编译以提高性能。 | 具有 FP8 块级量化的大规模 EP 部署。 |
| `cutlass` | 基于 CUTLASS 的高效 GEMM 后端。 | 具有 CUTLASS 支持的 NVIDIA 架构。 |
| `flashinfer_trtllm` | FlashInfer 与 TensorRT-LLM 集成，用于加速 MoE 计算，支持 FP4 通信运算符和高性能 GEMM。 | 具有 TRT-LLM 的 Blackwell。 |
| `flashinfer_cutlass` | FlashInfer 与 CUTLASS 结合，用于 MoE 层中的高性能分组 GEMM，高效处理 FP4/FP8 量化。 | 具有 FP4/FP8 模型的 Blackwell。 |
| `flashinfer_mxfp4` | FlashInfer 变体，针对 MoE 运行器中的 MXFP4（混合 FP4）量化进行了优化，专注于内存高效的低精度推理。 | 具有 MXFP4 的低精度模型。 |
| `flashinfer_cutedsl` | FlashInfer 带有自定义 DSL，用于灵活高效的 MoE 内核生成，与 ModelOpt FP4 量化集成。 | 具有 NVFP4 的低精度模型。 |

### Examples

Launch with DeepEP and DeepGEMM for DeepSeek-V3:

```bash
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --moe-a2a-backend deepep --moe-runner-backend deep_gemm --tp 8 --ep 8
```

## Extensible EP Framework

SGLang's EP framework provides modular abstractions for easy integration of custom kernels, backends, and optimizations. It decouples the MoE forward pass into stages (dispatch → pre-permute → core runner → post-permute → combine), enabling seamless extensions without refactoring core logic.

### Framework Overview

The framework centers on `FusedMoE` as the unified entry point for a single, extensible structure. Key components include:
- **Dispatcher**: Manages dispatch/combine for backends like DeepEP (implements `BaseDispatcher` subclasses).
- **MoeRunner**: Orchestrates grouped-GEMM execution via `MoeRunnerCore` implementations (e.g., `TritonRunnerCore`).
- **PermuteMethodPool**: Auto-registers layout conversions (e.g., pre/post-permute via `register_pre_permute` and `register_post_permute` for dynamic modes, or `register_fused_func` for static, torch.compile-compatible fused operations).
- **TopK Router**: Backend-agnostic expert selection.

This design supports multiple backends via `--moe-a2a-backend` and `--moe-runner-backend`, with quantization integrated through a standardized `apply()` method. The computation flow ensures modularity:

```
[input_hidden_states]
          |
          v
     TopK.forward -> select_experts / triton_kernels.routing / bypass
          |
          v
     [TopKOutput]
          |
          v
   FusedMoE.forward -> Dispatcher.dispatch -> DeepEP / bypass
          |                     |
          |                     v
          |              [DispatchOutput]
          |                     |
          |                     v
          |             quant_method.apply -> MoeRunner.forward
          |                     |              |
          |                     |              v
          |                     | pre-permute + grouped_gemm + post-permute
          |                     |              |
          |                     |--------------
          |                     v
          |               [CombineInput]
          |                     |
          |                     v
          |            Dispatcher.combine -> DeepEP / bypass
          |                     |
          |---------------------
          v
[final_hidden_states]
```

For details, see the [MoE Refactor Roadmap](https://github.com/sgl-project/sglang/issues/8715).

### Implementing New Backends

To add a new backend:
1. For a new all-to-all dispatcher, implement a `BaseDispatcher` subclass with `dispatch` and `combine` methods.
2. For a new MoE runner backend, define a `MoeRunnerCore` subclass for core operations (e.g., grouped GEMMs).
3. Define new input/output formats for the dispatcher or model runner (e.g., `RunnerInput`, `RunnerOutput`).
4. Register permute/unpermute methods to ensure compatibility:
   - **Fused Mode** (static, torch.compile-compatible): Use `register_fused_func` for end-to-end operations.
   - **Permute Mode** (dynamic): Register `register_pre_permute` and `register_post_permute` for flexible layouts.

See the [MoE Refactor Implementation PR](https://github.com/sgl-project/sglang/pull/9269) for full changes, including type hints and config expansions.

### Examples

For an example implementation, see [moe_runner/triton.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/triton.py), which demonstrates Triton-based grouped GEMMs with registered fused and permutation functions.

## Computation and Communication Overlap

SGLang's EP employs advanced overlap techniques to hide communication latency behind computation, maximizing GPU utilization in MoE layers.

### Two-Batch Overlap (TBO)

TBO splits requests into micro-batches, interleaving attention computation with dispatch/combine operations. Yield points in the execution graph allow pausing for overlaps, increasing overall throughput without peak memory spikes:

**中文对照**：## 计算和通信重叠

SGLang 的 EP 采用先进的重叠技术来隐藏计算背后的通信延迟，最大化 MoE 层中的 GPU 利用率。

### 双批次重叠（TBO）

TBO 将请求分成微批次，将注意力计算与分发/合并操作交错执行。执行图中的让出点允许暂停以进行重叠，在不产生峰值内存激增的情况下增加整体吞吐量：

```python
operations = [
    self._forward_attn,
    YieldOperation(),  # Overlap with dispatch of prior micro-batch
    self._forward_dispatch,
    self._forward_mlp,
    YieldOperation(),  # Overlap with combine
    self._forward_combine,
]
```

Users need to specify `--enable-two-batch-overlap` to unlock up to 2x throughput. For details, see the [Large-Scale EP Blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/#two-batch-overlap).

### Single-Batch Overlap (SBO)

SGLang introduces a dispatcher-hook system for Single-Batch Overlap (SBO), enabling the overlap of operations within a single batch—such as shared experts computation with communication—while decentralizing logic to enhance modularity. These hooks execute before and after the `dispatch` and `combine` operations without modifying core MoE modules. This design simplifies interfaces, reduces coupling, and improves extensibility. For implementation details and an example of overlapping shared experts with DeepEP's combine operation, refer to [PR #13327](https://github.com/sgl-project/sglang/pull/13327). Users can set `--enable-single-batch-overlap` to enable this feature.

**中文对照**：用户需要指定 `--enable-two-batch-overlap` 以解锁最高 2 倍的吞吐量。详情请参阅[大规模 EP 博客](https://lmsys.org/blog/2025-05-05-large-scale-ep/#two-batch-overlap)。

### 单批次重叠（SBO）

SGLang 为单批次重叠（SBO）引入了分发器钩子系统，启用单批次内的操作重叠——如共享专家计算与通信——同时分散逻辑以增强模块化。这些钩子在 `dispatch` 和 `combine` 操作之前和之后执行，而不修改核心 MoE 模块。此设计简化了接口，减少了耦合，并提高了可扩展性。关于实现细节以及将共享专家与 DeepEP 的 combine 操作重叠的示例，请参阅 [PR #13327](https://github.com/sgl-project/sglang/pull/13327)。用户可以设置 `--enable-single-batch-overlap` 来启用此功能。


## Workload Balancer

SGLang integrates the [Expert Parallelism Load Balancer (EPLB)](https://github.com/deepseek-ai/EPLB) from DeepSeek to address routing imbalances in MoE models. By analyzing expert activation statistics, EPLB computes an optimal expert arrangement, strategically placing or replicating experts to minimize GPU utilization variance, reduce idle cycles, and enhance scalability.

To enable EPLB, use the flags `--enable-eplb`. For optimal performance, increase batch sizes to stabilize activation statistics and configure periodic rebalancing (e.g., every 1000 requests) to adapt to evolving workloads. Simulations demonstrate significant improvements in load balancedness (ratio of mean to max computation time), correlating strongly with throughput gains.

For more details, refer to the [EPLB Section in the Large-Scale EP Blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/#expert-parallelism-load-balancer) and the [EPLB Repository](https://github.com/deepseek-ai/eplb).

**中文对照**：## 工作负载均衡器

SGLang 集成了来自 DeepSeek 的[专家并行负载均衡器（EPLB）](https://github.com/deepseek-ai/EPLB)以解决 MoE 模型中的路由不平衡问题。通过分析专家激活统计，EPLB 计算最优的专家安排，战略性地放置或复制专家以最小化 GPU 利用率差异，减少空闲周期，并增强可扩展性。

要启用 EPLB，请使用标志 `--enable-eplb`。为了获得最佳性能，增加批量大小以稳定激活统计，并配置定期重新平衡（例如每 1000 个请求）以适应不断变化的工作负载。模拟显示了负载均衡性的显著改善（平均计算时间与最大计算时间的比率），与吞吐量提升密切相关。

有关更多详细信息，请参阅[大规模 EP 博客中的 EPLB 部分](https://lmsys.org/blog/2025-05-05-large-scale-ep/#expert-parallelism-load-balancer)和 [EPLB 仓库](https://github.com/deepseek-ai/eplb)。


## EP with Spectulative Decoding


When utilizing speculative decoding with MTP on MoE architectures, use the `--speculative-moe-runner-backend` and `--speculative-moe-a2a-backend` arguments to customize the MoE layer behavior for the draft model. While they default to the target model's settings, users can differentiate them for varying precisions between target and draft models.

For model like `nvidia/DeepSeek-R1-0528-NVFP4-v2`, the target model uses NVFP4 precision while the draft model uses BF16. To apply `flashinfer_trtllm` kernel for target MoE layer while falling back to triton fused MoE kernel for draft MoE layer, users can set the arguments as follows:

**中文对照**：## 带有推测解码的 EP


在 MoE 架构上使用 MTP 进行推测解码时，使用 `--speculative-moe-runner-backend` 和 `--speculative-moe-a2a-backend` 参数来自定义草稿模型的 MoE 层行为。虽然它们默认使用目标模型的设置，但用户可以针对目标和草稿模型之间不同的精度进行区分。

对于像 `nvidia/DeepSeek-R1-0528-NVFP4-v2` 这样的模型，目标模型使用 NVFP4 精度，而草稿模型使用 BF16。要对目标 MoE 层应用 `flashinfer_trtllm` 内核，同时对草稿 MoE 层回退到 triton 融合 MoE 内核，用户可以按如下方式设置参数：
```
...
--moe-runner-backend flashinfer_trtllm \
--speculative-moe-runner-backend triton \
...
```


## Ascend NPU Guidance


### Guidance on SGLang configuration in Ascend NPU
- `--moe-a2a-backend` only supports `deepep` and `ascend_fuseep` backends,
  - `deepep`: The mechanism is consistent with the above description.
  - `ascend_fuseep`: Offer a large fused operator which integrates all operations between dispatch and combine to boost MoE computation. Only used for decode stage in PD Disaggregation Mode.
- `--moe-runner-backend` parameter does not need to be configured.
- `--deepep-mode`:
  - In PD mixed mode, please set `--deepep-mode auto`.
  - In PD Disaggregation Mode, prefill instance sets `--deepep-mode normal`, and decode instance sets `--deepep-mode low_latency`.


### DeepEP Ascend Introduction

DeepEP Ascend is the adapted version of the DeepEP communication library for Huawei Ascend NPUs, specifically designed for Mixture-of-Experts (MoE) model Expert Parallelism (EP).
It supports the Ant-moving Function (Split the sequence length into rounds for streaming batch transmission) to optimize the buffer size occupied during collective communication in prefill stage, especially for long sequences.

**中文对照**：## Ascend NPU 指导


### Ascend NPU 中的 SGLang 配置指导
- `--moe-a2a-backend` 仅支持 `deepep` 和 `ascend_fuseep` 后端，
  - `deepep`：机制与上述描述一致。
  - `ascend_fuseep`：提供一个大型融合运算符，整合了 dispatch 和 combine 之间的所有操作以提升 MoE 计算。仅在 PD 分离模式的解码阶段使用。
- `--moe-runner-backend` 参数不需要配置。
- `--deepep-mode`：
  - 在 PD 混合模式中，请设置 `--deepep-mode auto`。
  - 在 PD 分离模式中，预填充实例设置 `--deepep-mode normal`，解码实例设置 `--deepep-mode low_latency`。


### DeepEP Ascend 简介

DeepEP Ascend 是适用于华为 Ascend NPU 的 DeepEP 通信库的适配版本，专为混合专家（MoE）模型的专家并行（EP）而设计。
它支持蚂蚁移动功能（将序列长度分成轮次进行流式批次传输）以优化预填充阶段集体通信期间占用的缓冲区大小，特别是对于长序列。

Ant-moving Function can be enabled for both the dispatch and combine phases via the following environment variables:
- `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS`: Enable ant-moving function in dispatch stage. Indicates the number of tokens transmitted per round on each rank, default 8192.
- `DEEPEP_NORMAL_LONG_SEQ_ROUND`: Enable ant-moving function in dispatch stage. Indicates the number of rounds transmitted on each rank, default 1.
- `DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ`: Enable ant-moving function in combine stage, default 0 (means disabled).

`DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS * DEEPEP_NORMAL_LONG_SEQ_ROUND` means input sequence length. When the input sequence length exceeds 8192, it is recommended to enable the ant-moving function in both dispatch and combine phase.

The environment variable `HCCL_BUFFSIZE` is used to configure the buffer size (MB) actually allocated. Its calculation formula is as follows:
```angular2html
# Enable Ant-moving Function
HCCL_BUFFSIZE >= 2 * (102MB + 4MB + DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS * (hidden_size + hidden_size + hidden_size) * topk) + PADDING_BUFFSIZE

# Disable Ant-moving Function
HCCL_BUFFSIZE >= 2 * (102MB + 4MB + TOTAL_SEQ_LEN * (hidden_size + hidden_size) * topk) + PADDING_BUFFSIZE
```
Wherein the parameters are described as follows:
- `hidden_size`: hidden size in model config.
- `topk`: The number of selected routing experts.
- `TOTAL_SEQ_LEN`: input sequence length.
- `PADDING_BUFFSIZE`: A value of 20 or greater is recommended.

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | `FusedMoE` 统一入口点：dispatch → permute → runner → combine 流水线 |
| `python/sglang/srt/layers/moe/token_dispatcher/base.py` | `BaseDispatcher` all-to-all通信后端（DeepEP、Mooncake、MORI）的抽象类 |
| `python/sglang/srt/layers/moe/moe_runner/base.py` | `MoeRunnerCore` grouped-GEMM后端（Triton、DeepGEMM、CUTLASS）的抽象类 |
| `python/sglang/srt/layers/moe/moe_runner/runner.py` | `MoeRunner` 协调器：pre-permute → core GEMM → post-permute |
| `python/sglang/srt/layers/moe/utils.py` | `PermuteMethodPool` 布局转换的自动注册（融合/动态模式） |
| `python/sglang/srt/distributed/parallel_state.py` | EP进程组管理和 `ep_size` 初始化 |

### 架构

```
[input_hidden_states]
       │
       ▼
  TopK Router ──▶ select experts (backend-agnostic)
       │
       ▼
  FusedMoE.forward
       │
       ├──▶ Dispatcher.dispatch (DeepEP / Mooncake / MORI / bypass)
       │         │
       │         ▼
       │    quant_method.apply ──▶ MoeRunner.forward
       │         │                    │
       │         │     pre-permute + grouped_gemm + post-permute
       │         │                    │
       │         ◀────────────────────┘
       │         │
       │    Dispatcher.combine (all-to-all gather)
       │         │
       ◀─────────┘
       ▼
[final_hidden_states]
```

### 关键代码逻辑

- **后端选择**: `--moe-a2a-backend` 选择dispatcher（DeepEP normal/low_latency、Mooncake、MORI、FlashInfer）；`--moe-runner-backend` 选择GEMM后端（Triton、DeepGEMM、CUTLASS、FlashInfer变体）
- **两批次重叠（TBO）**: 通过执行图中的 `YieldOperation()` 点将注意力与dispatch/combine交错执行
- **单批次重叠（SBO）**: Dispatcher-hook系统将共享专家计算与通信重叠
- **EPLB集成**: 专家并行负载均衡器根据激活统计数据重新平衡专家放置

### 集成要点

- **服务器标志**: `--tp`、`--ep`、`--moe-a2a-backend`、`--moe-runner-backend`、`--deepep-mode`、`--enable-two-batch-overlap`、`--enable-single-batch-overlap`、`--enable-eplb`
- **推测解码**: `--speculative-moe-runner-backend` 和 `--speculative-moe-a2a-backend` 用于draft模型MoE配置
- **可扩展性**: 为新的all-to-all后端实现 `BaseDispatcher` 子类，为新的GEMM后端实现 `MoeRunnerCore` 子类
