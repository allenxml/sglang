# DeepSeek V3/V3.1/R1 Usage

SGLang provides many optimizations specifically designed for the DeepSeek models, making it the inference engine recommended by the official [DeepSeek team](https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#62-inference-with-sglang-recommended) from Day 0.

**中文对照**：SGLang 提供了许多专门针对 DeepSeek 模型优化的功能，使其成为官方 DeepSeek 团队从第一天起就推荐使用的推理引擎。

This document outlines current optimizations for DeepSeek.
For an overview of the implemented features see the completed [Roadmap](https://github.com/sgl-project/sglang/issues/2591).

**中文对照**：本文档概述了 DeepSeek 当前的优化措施。已实现功能的概述请参阅完整的[路线图](https://github.com/sgl-project/sglang/issues/2591)。

## Launch DeepSeek V3.1/V3/R1 with SGLang

**中文对照**：使用 SGLang 启动 DeepSeek V3.1/V3/R1

To run DeepSeek V3.1/V3/R1 models, the recommended settings are as follows:

**中文对照**：运行 DeepSeek V3.1/V3/R1 模型的推荐设置如下：

| Weight Type | Configuration |
|------------|-------------------|
| **Full precision [FP8](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)**<br>*(recommended)* | 8 x H200 |
| | 8 x B200 |
| | 8 x MI300X |
| | 2 x 8 x H100/800/20 |
| | Xeon 6980P CPU |
| **Full precision ([BF16](https://huggingface.co/unsloth/DeepSeek-R1-0528-BF16))** (upcast from original FP8) | 2 x 8 x H200 |
| | 2 x 8 x MI300X |
| | 4 x 8 x H100/800/20 |
| | 4 x 8 x A100/A800 |
| **Quantized weights ([INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8))** | 16 x A100/800 |
| | 32 x L40S |
| | Xeon 6980P CPU |
| | 4 x Atlas 800I A3 |
| **Quantized weights ([W4A8](https://huggingface.co/novita/Deepseek-R1-0528-W4AFP8))** | 8 x H20/100, 4 x H200 |
| **Quantized weights ([AWQ](https://huggingface.co/QuixiAI/DeepSeek-R1-0528-AWQ))** | 8 x H100/800/20 |
| | 8 x A100/A800 |
| **Quantized weights ([MXFP4](https://huggingface.co/amd/DeepSeek-R1-MXFP4-Preview))** | 8, 4 x MI355X/350X |
| **Quantized weights ([NVFP4](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4-v2))** | 8, 4 x B200 |

<style>
.md-typeset__table {
  width: 100%;
}

.md-typeset__table table {
  border-collapse: collapse;
  margin: 1em 0;
  border: 2px solid var(--md-typeset-table-color);
  table-layout: fixed;
}

.md-typeset__table th {
  border: 1px solid var(--md-typeset-table-color);
  border-bottom: 2px solid var(--md-typeset-table-color);
  background-color: var(--md-default-bg-color--lighter);
  padding: 12px;
}

.md-typeset__table td {
  border: 1px solid var(--md-typeset-table-color);
  padding: 12px;
}

.md-typeset__table tr:nth-child(2n) {
  background-color: var(--md-default-bg-color--lightest);
}
</style>

```{important}
The official DeepSeek V3 is already in FP8 format, so you should not run it with any quantization arguments like `--quantization fp8`.
```

Detailed commands for reference:

- [8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#using-docker-recommended)
- [4 x B200, 8 x B200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-one-b200-node)
- [8 x MI300X](../platforms/amd_gpu.md#running-deepseek-v3)
- [2 x 8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h208-nodes)
- [4 x 8 x A100](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes)
- [8 x A100 (AWQ)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-8-a100a800-with-awq-quantization)
- [16 x A100 (INT8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-16-a100a800-with-int8-quantization)
- [32 x L40S (INT8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-32-l40s-with-int8-quantization)
- [Xeon 6980P CPU](../platforms/cpu_server.md#example-running-deepseek-r1)
- [4 x Atlas 800I A3 (int8)](../platforms/ascend_npu_deepseek_example.md#running-deepseek-with-pd-disaggregation-on-4-x-atlas-800i-a3)

### Download Weights

**中文对照**：下载权重

If you encounter errors when starting the server, ensure the weights have finished downloading. It's recommended to download them beforehand or restart multiple times until all weights are downloaded. Please refer to [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base#61-inference-with-deepseek-infer-demo-example-only) official guide to download the weights.

**中文对照**：如果在启动服务器时遇到错误，请确保权重已下载完成。建议提前下载或多次重启直到所有权重下载完成。请参阅 [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base#61-inference-with-deepseek-infer-demo-example-only) 官方指南下载权重。

### Launch with one node of 8 x H200

**中文对照**：使用 8 个 H200 单节点启动

Please refer to [the example](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#installation--launch).

**中文对照**：请参阅[示例](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#installation--launch)。

### Running examples on Multi-Node

**中文对照**：多节点运行示例

- [Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP](https://lmsys.org/blog/2025-06-16-gb200-part-1/) ([Part I](https://lmsys.org/blog/2025-06-16-gb200-part-1/), [Part II](https://lmsys.org/blog/2025-09-25-gb200-part-2/)) - Comprehensive guide on GB200 optimizations.

**中文对照**：[在 GB200 NVL72 上部署 DeepSeek：PD 和大规模 EP](https://lmsys.org/blog/2025-06-16-gb200-part-1/)（[第一部分](https://lmsys.org/blog/2025-06-16-gb200-part-1/)、[第二部分](https://lmsys.org/blog/2025-09-25-gb200-part-2/)）- GB200 优化的综合指南。

- [Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs](https://lmsys.org/blog/2025-05-05-deepseek-pd-ep/) - Guide on PD disaggregation and large-scale EP.

**中文对照**：[在 96 个 H100 GPU 上使用 PD 分解和大规模专家并行部署 DeepSeek](https://lmsys.org/blog/2025-05-05-deepseek-pd-ep/) - PD 分解和大规模 EP 指南。

- [Serving with two H20*8 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h208-nodes).

**中文对照**：[使用两个 H20*8 节点服务](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h208-nodes)。

- [Best Practices for Serving DeepSeek-R1 on H20](https://lmsys.org/blog/2025-09-26-sglang-ant-group/) - Comprehensive guide on H20 optimizations, deployment and performance.

**中文对照**：[在 H20 上服务 DeepSeek-R1 的最佳实践](https://lmsys.org/blog/2025-09-26-sglang-ant-group/) - H20 优化、部署和性能的综合指南。

- [Serving with two H200*8 nodes and docker](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker).

**中文对照**：[使用两个 H200*8 节点和 Docker 服务](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker)。

- [Serving with four A100*8 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes).

**中文对照**：[使用四个 A100*8 节点服务](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes)。

## Optimizations

**中文对照**：优化

### Multi-head Latent Attention (MLA) Throughput Optimizations

**中文对照**：多头潜在注意力 (MLA) 吞吐量优化

**Description**: [MLA](https://arxiv.org/pdf/2405.04434) is an innovative attention mechanism introduced by the DeepSeek team, aimed at improving inference efficiency. SGLang has implemented specific optimizations for this, including:

**中文对照**：[MLA](https://arxiv.org/pdf/2405.04434) 是 DeepSeek 团队引入的一种创新注意力机制，旨在提高推理效率。SGLang 为此实现了特定的优化，包括：

- **Weight Absorption**: By applying the associative law of matrix multiplication to reorder computation steps, this method balances computation and memory access and improves efficiency in the decoding phase.

**中文对照**：**权重吸收**：通过应用矩阵乘法的结合律重新排列计算步骤，该方法平衡了计算和内存访问，提高了解码阶段的效率。

- **MLA Attention Backends**: Currently SGLang supports different optimized MLA attention backends, including [FlashAttention3](https://github.com/Dao-AILab/flash-attention), [Flashinfer](https://docs.flashinfer.ai/api/attention.html#flashinfer-mla), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [CutlassMLA](https://github.com/sgl-project/sglang/pull/5390), **TRTLLM MLA** (optimized for Blackwell architecture), and [Triton](https://github.com/triton-lang/triton) backends. The default FA3 provides good performance across wide workloads.

**中文对照**：**MLA 注意力后端**：SGLang 目前支持不同的优化 MLA 注意力后端，包括 [FlashAttention3](https://github.com/Dao-AILab/flash-attention)、[Flashinfer](https://docs.flashinfer.ai/api/attention.html#flashinfer-mla)、[FlashMLA](https://github.com/deepseek-ai/FlashMLA)、[CutlassMLA](https://github.com/sgl-project/sglang/pull/5390)、**TRTLLM MLA**（针对 Blackwell 架构优化）和 [Triton](https://github.com/triton-lang/triton) 后端。默认的 FA3 在各种工作负载上提供良好的性能。

- **FP8 Quantization**: W8A8 FP8 and KV Cache FP8 quantization enables efficient FP8 inference. Additionally, we have implemented Batched Matrix Multiplication (BMM) operator to facilitate FP8 inference in MLA with weight absorption.

**中文对照**：**FP8 量化**：W8A8 FP8 和 KV 缓存 FP8 量化可实现高效的 FP8 推理。此外，我们还实现了批量矩阵乘法（BMM）算子，以促进 MLA 中带权重吸收的 FP8 推理。

- **CUDA Graph & Torch.compile**: Both MLA and Mixture of Experts (MoE) are compatible with CUDA Graph and Torch.compile, which reduces latency and accelerates decoding speed for small batch sizes.

**中文对照**：**CUDA Graph 和 Torch.compile**：MLA 和混合专家（MoE）都与 CUDA Graph 和 Torch.compile 兼容，这可以减少延迟并加速小批量的解码速度。

- **Chunked Prefix Cache**: Chunked prefix cache optimization can increase throughput by cutting prefix cache into chunks, processing them with multi-head attention and merging their states. Its improvement can be significant when doing chunked prefill on long sequences. Currently this optimization is only available for FlashAttention3 backend.

**中文对照**：**分块前缀缓存**：分块前缀缓存优化可以通过将前缀缓存分块来处理增加吞吐量，使用多头注意力处理它们并合并它们的状态。在对长序列进行分块预填充时，其改进可能非常显著。目前此优化仅适用于 FlashAttention3 后端。

Overall, with these optimizations, we have achieved up to **7x** acceleration in output throughput compared to the previous version.

**中文对照**：总体而言，通过这些优化，我们实现了相比上一版本高达 **7倍** 的输出吞吐量加速。

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_3/deepseek_mla.svg" alt="Multi-head Latent Attention for DeepSeek Series Models">
</p>

**Usage**: MLA optimization is enabled by default.

**中文对照**：**用法**：MLA 优化默认启用。

**Reference**: Check [Blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) and [Slides](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/lmsys_1st_meetup_deepseek_mla.pdf) for more details.

**中文对照**：**参考**：查看[博客](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations)和[幻灯片](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/lmsys_1st_meetup_deepseek_mla.pdf)了解更多详情。

### Data Parallelism Attention

**中文对照**：数据并行注意力

**Description**: This optimization involves data parallelism (DP) for the MLA attention mechanism of DeepSeek Series Models, which allows for a significant reduction in the KV cache size, enabling larger batch sizes. Each DP worker independently handles different types of batches (prefill, decode, idle), which are then synchronized before and after processing through the Mixture-of-Experts (MoE) layer. If you do not use DP attention, KV cache will be duplicated among all TP ranks.

**中文对照**：**描述**：此优化涉及 DeepSeek 系列模型的 MLA 注意力机制的数据并行（DP），这可以显著减少 KV 缓存大小，从而支持更大的批次大小。每个 DP 工作进程独立处理不同类型的批次（预填充、解码、空闲），然后在通过混合专家（MoE）层处理之前和之后进行同步。如果不使用 DP 注意力，KV 缓存将在所有 TP 秩之间复制。

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/dp_attention.svg" alt="Data Parallelism Attention for DeepSeek Series Models">
</p>

With data parallelism attention enabled, we have achieved up to **1.9x** decoding throughput improvement compared to the previous version.

**中文对照**：启用数据并行注意力后，我们实现了相比上一版本高达 **1.9倍** 的解码吞吐量提升。

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/deepseek_coder_v2.svg" alt="Data Parallelism Attention Performance Comparison">
</p>

**Usage**:
- Append `--enable-dp-attention --tp 8 --dp 8` to the server arguments when using 8 H200 GPUs. This optimization improves peak throughput in high batch size scenarios where the server is limited by KV cache capacity.
- DP and TP attention can be flexibly combined. For example, to deploy DeepSeek-V3/R1 on 2 nodes with 8 H100 GPUs each, you can specify `--enable-dp-attention --tp 16 --dp 2`. This configuration runs attention with 2 DP groups, each containing 8 TP GPUs.

```{caution}
Data parallelism attention is not recommended for low-latency, small-batch use cases. It is optimized for high-throughput scenarios with large batch sizes.
```

**Reference**: Check [Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models).

**中文对照**：**参考**：查看[博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models)了解更多详情。

### Multi-Node Tensor Parallelism

**Description**: For users with limited memory on a single node, SGLang supports serving DeepSeek Series Models, including DeepSeek V3, across multiple nodes using tensor parallelism. This approach partitions the model parameters across multiple GPUs or nodes to handle models that are too large for one node's memory.

**中文对照**：**描述**：对于单个节点内存受限的用户，SGLang 支持使用张量并行在多个节点上提供 DeepSeek 系列模型（包括 DeepSeek V3）的服务。这种方法将模型参数分区到多个 GPU 或节点上，以处理单个节点内存无法容纳的模型。

**Usage**: Check [here](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208) for usage examples.

**中文对照**：**用法**：查看[此处](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208)的用法示例。

### Block-wise FP8

**中文对照**：分块 FP8

**Description**: SGLang implements block-wise FP8 quantization with two key optimizations:

**中文对照**：**描述**：SGLang 实现分块 FP8 量化，包含两个关键优化：

- **Activation**: E4M3 format using per-token-per-128-channel sub-vector scales with online casting.

**中文对照**：**激活**：E4M3 格式，使用每令牌每 128 通道子向量缩放和在线转换。

- **Weight**: Per-128x128-block quantization for better numerical stability.

**中文对照**：**权重**：每 128x128 块量化以获得更好的数值稳定性。

- **DeepGEMM**: The [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernel library optimized for FP8 matrix multiplications.

**中文对照**：**DeepGEMM**：[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 是针对 FP8 矩阵乘法优化的内核库。

**Usage**: The activation and weight optimization above are turned on by default for DeepSeek V3 models. DeepGEMM is enabled by default on NVIDIA Hopper/Blackwell GPUs and disabled by default on other devices. DeepGEMM can also be manually turned off by setting the environment variable `SGLANG_ENABLE_JIT_DEEPGEMM=0`.

**中文对照**：**用法**：上述激活和权重优化对于 DeepSeek V3 模型默认启用。DeepGEMM 在 NVIDIA Hopper/Blackwell GPU 上默认启用，在其他设备上默认禁用。DeepGEMM 也可以通过设置环境变量 `SGLANG_ENABLE_JIT_DEEPGEMM=0` 手动关闭。

```{tip}
Before serving the DeepSeek model, precompile the DeepGEMM kernels to improve first-run performance. The precompilation process typically takes around 10 minutes to complete.
```

```bash
python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```

### Multi-token Prediction

**中文对照**：多令牌预测

**Description**: SGLang implements DeepSeek V3 Multi-Token Prediction (MTP) based on [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding). With this optimization, the decoding speed can be improved by **1.8x** for batch size 1 and **1.5x** for batch size 32 respectively on H200 TP8 setting.

**中文对照**：**描述**：SGLang 基于 [EAGLE 推测解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding) 实现了 DeepSeek V3 多令牌预测（MTP）。使用此优化，在 H200 TP8 设置下，批次大小为 1 时解码速度可提高 **1.8倍**，批次大小为 32 时可提高 **1.5倍**。

**Usage**:

**中文对照**：**用法**：

Add `--speculative-algorithm EAGLE`. Other flags, like `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` are optional. For example:

**中文对照**：添加 `--speculative-algorithm EAGLE`。其他标志如 `--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 是可选的。例如：
```
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --speculative-algorithm EAGLE \
  --trust-remote-code \
  --tp 8
```
- The default configuration for DeepSeek models is `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`. The best configuration for `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` can be searched with [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) script for given batch size. The minimum configuration is `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`, which can achieve speedup for larger batch sizes.
- Most MLA attention backends fully support MTP usage. See [MLA Backends](../advanced_features/attention_backend.md#mla-backends) for details.

```{note}
To enable DeepSeek MTP for large batch sizes (>48), you need to adjust some parameters (Reference [this discussion](https://github.com/sgl-project/sglang/issues/4543#issuecomment-2737413756)):
- Adjust `--max-running-requests` to a larger number. The default value is `48` for MTP. For larger batch sizes, you should increase this value beyond the default value.
- Set `--cuda-graph-bs`. It's a list of batch sizes for cuda graph capture. The [default captured batch sizes for speculative decoding](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L888-L895) is 48. You can customize this by including more batch sizes.
```

```{tip}
To enable the experimental overlap scheduler for EAGLE speculative decoding, set the environment variable `SGLANG_ENABLE_SPEC_V2=1`. This can improve performance by enabling overlap scheduling between draft and verification stages.
```


### Reasoning Content for DeepSeek R1 & V3.1

**中文对照**：DeepSeek R1 和 V3.1 的推理内容

See [Reasoning Parser](https://docs.sglang.io/advanced_features/separate_reasoning.html) and [Thinking Parameter for DeepSeek V3.1](https://docs.sglang.io/basic_usage/openai_api_completions.html#Example:-DeepSeek-V3-Models).

**中文对照**：请参阅[推理解析器](https://docs.sglang.io/advanced_features/separate_reasoning.html)和 [DeepSeek V3.1 的思考参数](https://docs.sglang.io/basic_usage/openai_api_completions.html#Example:-DeepSeek-V3-Models)。


### Function calling for DeepSeek Models

**中文对照**：DeepSeek 模型的函数调用

Add arguments `--tool-call-parser deepseekv3` and `--chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja`(recommended) to enable this feature. For example (running on 1 * H20 node):

**中文对照**：添加参数 `--tool-call-parser deepseekv3` 和 `--chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja`（推荐）以启用此功能。例如（在 1 个 H20 节点上运行）：

```
python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3-0324 \
  --tp 8 \
  --port 30000 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.9 \
  --tool-call-parser deepseekv3 \
  --chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja
```

Sample Request:

**中文对照**：示例请求：

```
curl "http://127.0.0.1:30000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{"temperature": 0, "max_tokens": 100, "model": "deepseek-ai/DeepSeek-V3-0324", "tools": [{"type": "function", "function": {"name": "query_weather", "description": "Get weather of a city, the user should supply a city first", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city, e.g. Beijing"}}, "required": ["city"]}}}], "messages": [{"role": "user", "content": "How'\''s the weather like in Qingdao today"}]}'
```

Expected Response

**中文对照**：预期响应

```
{"id":"6501ef8e2d874006bf555bc80cddc7c5","object":"chat.completion","created":1745993638,"model":"deepseek-ai/DeepSeek-V3-0324","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"id":"0","index":null,"type":"function","function":{"name":"query_weather","arguments":"{\"city\": \"Qingdao\"}"}}]},"logprobs":null,"finish_reason":"tool_calls","matched_stop":null}],"usage":{"prompt_tokens":116,"total_tokens":138,"completion_tokens":22,"prompt_tokens_details":null}}

```
Sample Streaming Request:

**中文对照**：示例流式请求：
```
curl "http://127.0.0.1:30000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{"temperature": 0, "max_tokens": 100, "model": "deepseek-ai/DeepSeek-V3-0324","stream":true,"tools": [{"type": "function", "function": {"name": "query_weather", "description": "Get weather of a city, the user should supply a city first", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city, e.g. Beijing"}}, "required": ["city"]}}}], "messages": [{"role": "user", "content": "How'\''s the weather like in Qingdao today"}]}'
```
Expected Streamed Chunks (simplified for clarity):

**中文对照**：预期的流式块（为清晰起见已简化）：
```
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"city"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\":\""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"Q"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"ing"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"dao"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\"}"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":null}}], "finish_reason": "tool_calls"}
data: [DONE]
```
The client needs to concatenate all arguments fragments to reconstruct the complete tool call:

**中文对照**：客户端需要连接所有参数片段以重建完整的工具调用：
```
{"city": "Qingdao"}
```

```{important}
1. Use a lower `"temperature"` value for better results.
2. To receive more consistent tool call results, it is recommended to use `--chat-template examples/chat_template/tool_chat_template_deepseekv3.jinja`. It provides an improved unified prompt.
```



### Thinking Budget for DeepSeek R1

**中文对照**：DeepSeek R1 的思考预算

In SGLang, we can implement thinking budget with `CustomLogitProcessor`.

**中文对照**：在 SGLang 中，我们可以使用 `CustomLogitProcessor` 实现思考预算。

Launch a server with `--enable-custom-logit-processor` flag on.

**中文对照**：使用 `--enable-custom-logit-processor` 标志启动服务器。

```
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --tp 8 --port 30000 --host 0.0.0.0 --mem-fraction-static 0.9 --disable-cuda-graph --reasoning-parser deepseek-r1 --enable-custom-logit-processor
```

Sample Request:

**中文对照**：示例请求：

```python
import openai
from rich.pretty import pprint
from sglang.srt.sampling.custom_logit_processor import DeepSeekR1ThinkingBudgetLogitProcessor


client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="*")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {
            "role": "user",
            "content": "Question: Is Paris the Capital of France?",
        }
    ],
    max_tokens=1024,
    extra_body={
        "custom_logit_processor": DeepSeekR1ThinkingBudgetLogitProcessor().to_str(),
        "custom_params": {
            "thinking_budget": 512,
        },
    },
)
pprint(response)
```



## FAQ

**中文对照**：常见问题解答

**Q: Model loading is taking too long, and I'm encountering an NCCL timeout. What should I do?**

**中文对照**：**问**：模型加载时间太长，遇到 NCCL 超时。我该怎么办？

A: If you're experiencing extended model loading times and an NCCL timeout, you can try increasing the timeout duration. Add the argument `--dist-timeout 3600` when launching your model. This will set the timeout to one hour, which often resolves the issue.

**中文对照**：**答**：如果遇到模型加载时间过长和 NCCL 超时，可以尝试增加超时持续时间。在启动模型时添加参数 `--dist-timeout 3600`。这会将超时设置为一小时，通常可以解决问题。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/deepseek_v2.py` | DeepSeek V3 模型架构：MLA 注意力 + MoE 层 |
| `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` | MLA 专用注意力核（FlashMLA、CutlassMLA、FA3），支持权重吸收 |
| `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | MoE 层，包含 FusedMoE 入口以支持专家并行 |
| `python/sglang/srt/managers/scheduler_dp_attn_mixin.py` | DP 注意力：每个 DP worker 独立维护 KV cache，仅在 MoE 前后同步 |
| `python/sglang/srt/speculative/eagle_worker.py` | EAGLE 推测解码，用于 MTP（多令牌预测） |
| `python/sglang/srt/function_call/deepseekv3_detector.py` | DeepSeek V3 函数调用格式检测和解析 |

### 关键代码逻辑

- **MLA 权重吸收**：重新排列矩阵乘法顺序，在解码阶段平衡计算和内存访问；在 MLA 注意力后端中实现
- **分块 FP8**：每 128x128 块权重量化 + 每令牌每 128 通道激活缩放；DeepGEMM 在 Hopper/Blackwell 上 JIT 编译
- **DP 注意力**：`--enable-dp-attention --tp 8 --dp 8` 使每个 DP worker 独立运行注意力，仅在 MoE 边界同步
- **MTP 推测解码**：`--speculative-algorithm EAGLE` 使用草稿 MTP 头实现 1.5-1.8 倍解码加速

### 集成要点

- **服务器参数**：`--tp`、`--dp`、`--ep`、`--enable-dp-attention`、`--speculative-algorithm EAGLE`、`--tool-call-parser deepseekv3`、`--reasoning-parser deepseek-r1`
- **环境变量**：`SGLANG_ENABLE_JIT_DEEPGEMM`、`SGLANG_ENABLE_SPEC_V2`
- **预编译**：`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8`
