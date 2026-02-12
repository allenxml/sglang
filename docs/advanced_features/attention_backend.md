# Attention Backend

SGLang supports a large variety of attention backends. Each of them has different pros and cons.
You can test them according to your needs.

**中文对照**：SGLang 支持多种注意力后端。每种后端都有不同的优缺点。您可以根据自己的需求进行测试。

```{important}
Selecting an optimal attention backend is crucial for maximizing your performance. Different backends excel in various scenarios, so choose based on your model, hardware, and use case. Not all backends are supported on all platforms and model architectures.

If you don't specify `--attention-backend`, SGLang makes a best effort to automatically select the most performant backend based on your hardware and model architecture.
```

**中文对照**：选择最优的注意力后端对于最大化性能至关重要。不同的后端在不同场景下表现出色，因此请根据您的模型、硬件和用例进行选择。并非所有后端都支持所有平台和模型架构。

如果您不指定 `--attention-backend`，SGLang 会尽最大努力根据您的硬件和模型架构自动选择性能最佳的后端。

## Support Matrix

The support matrix is split into two parts: MHA (standard attention) and MLA (multi-head latent attention). For an explanation of the key differences between MHA and MLA, please see the [SGLang documentation on DeepSeek MLA](../basic_usage/deepseek_v3.md#multi-head-latent-attention-mla-throughput-optimizations) and the original [DeepSeek MLA paper](https://arxiv.org/pdf/2405.04434).

**中文对照**：支持矩阵分为两个部分：MHA（标准注意力）和 MLA（多头潜在注意力）。关于 MHA 和 MLA 之间主要区别的解释，请参阅 [SGLang DeepSeek MLA 文档](../basic_usage/deepseek_v3.md#multi-head-latent-attention-mla-throughput-optimizations) 和原始的 [DeepSeek MLA 论文](https://arxiv.org/pdf/2405.04434)。

### MHA Backends

| **Backend**                     | **Page Size > 1 (native)** | **FP8 KV Cache** | **FP4 KV Cache** | **Spec topk=1** | **Spec topk>1** | **Sliding Window** | **MultiModal** |
|---------------------------------|-----------------------------|------------------|-----------------|-----------------|-----------------|--------------------|----------------|
| **FlashInfer**                  | ✅                          | ✅               | ❌              | ✅              | ✅              | ✅                 | ❌             |
| **FA3 (FlashAttention 3)**      | ✅                          | ✅               | ❌              | ✅              | ✅              | ✅                 | ✅             |
| **FA4 (FlashAttention 4)**      | 128                         | ❌               | ✅              | ❌              | ❌              | ❌                 | ✅             |
| **Triton**                      | ❌                          | ❌               | ✅              | ✅              | ✅              | ✅                 | ✅             |
| **Torch Native (SDPA)**         | ❌                          | ✅               | ✅              | ❌              | ❌              | ❌                 | ✅             |
| **FlexAttention (PyTorch)**     | ❌                          | ❌               | ✅              | ❌              | ❌              | ❌                 | ❌             |
| **TRTLLM MHA**                  | 16, 32 or 64                | ✅               | ✅              | ✅              | ❌              | ✅                 | ❌             |
| **Dual Chunk FlashAttention**   | ✅                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |
| **AITER (ROCm)**                | ✅                          | ✅               | ❌              | ✅              | ✅              | ❌                 | ✅             |
| **Wave (ROCm)**                 | ✅                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |
| **Ascend (NPU)**                | ✅                          | ❌               | ❌              | ✅              | ❌              | ❌                 | ✅             |
| **Intel XPU**                   | ✅                          | ❌               | ❌              | ❌              | ❌              | ✅                 | ❌             |
| **Intel AMX (CPU)**             | ❌                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |

### MLA Backends

| **Backend**                | **Native Page Sizes**     | **FP8 KV Cache** | **FP4 KV Cache** | **Chunked Prefix Cache** | **Spec topk=1** | **Spec topk>1** |
|----------------------------|---------------------------|------------------|------------------|--------------------------|-----------------|-----------------|
| **FlashInfer MLA**         | 1                         | ❌               | ✅               | ✅                       | ✅              | ❌              |
| **FlashMLA**               | 64                        | ✅               | ✅               | ✅                       | ✅              | ❌              |
| **Cutlass MLA**            | 128                       | ✅               | ✅               | ✅                       | ✅              | ❌              |
| **TRTLLM MLA (Blackwell)** | 32 or 64                  | ✅               | ✅               | ✅                       | ✅              | ❌              |
| **FA3 (FlashAttention 3)** | n/a                       | ❌               | ❌               | ✅                       | ✅              | ⚠️ (page_size=1 only) |
| **Triton**                 | n/a                       | ❌               | ❌               | ❌                       | ✅              | ⚠️ (page_size=1 only) |
| **FA4**                    | 1                         | ❌               | ✅               | ❌                       | ❌              | ❌              |
| **Ascend MLA (NPU)**       | 128                       | ❌               | ❌               | ❌                       | ❌              | ❌              |

```{note}
Multimodal attention is selected by `--mm-attention-backend`. The "MultiModal" column indicates whether a corresponding multimodal implementation exists for that backend family.
```

```{note}
- FlashAttention 4 is prefill-only for now.
- NSA is specifically designed for [DeepSeek V3.2 DSA](https://lmsys.org/blog/2025-09-29-deepseek-V32/).
```

```{note}
For the KV4 FA4 scenario, FA4 requires using a different --decode-attention-backend to run. Except for trtllm_mha being incompatible with FA4, all other decode backends behave as shown in the table.
```

```{tip}
Speculative decoding topk: `topk` is the number of draft tokens sampled per step from the draft model. `topk = 1` follows classic EAGLE; `topk > 1` explores multiple branches and requires backend support in both draft and verification paths.
```

```{tip}
Page size controls how many tokens are grouped into a KV cache block. For the prefix cache to take effect, the number of tokens must fill at least one complete page. For example, if your prompt is only 32 tokens and `page_size = 64`, it won't fill a complete page and cannot be matched in the prefix cache (pages cannot be padded). With 65 tokens and `page_size = 64`, only the first page of 64 tokens will be cached and matched; the remaining 1 token is discarded. Use `page_size = 1` for maximum prefix reuse (token-level matching).
```

Many backends that do not natively operate on pages can emulate `page_size > 1` at the wrapper layer by expanding page tables to per-token indices. The "Page Size > 1 (native)" column indicates true in-kernel paging. Some backends require fixed native page sizes and cannot be reduced/emulated differently: TRTLLM MHA (16/32/64), TRTLLM MLA (32/64), FlashMLA (64), Cutlass MLA (128), Ascend (128).

**中文对照**：许多不原生在页面上操作的后端可以通过将页表扩展为每个令牌的索引在包装层模拟 `page_size > 1`。"Page Size > 1 (native)" 列表示真正的内核分页。有些后端需要固定的原生页面大小，不能减少或以不同方式模拟：TRTLLM MHA (16/32/64)、TRTLLM MLA (32/64)、FlashMLA (64)、Cutlass MLA (128)、Ascend (128)。

MLA page-size constraints:
- FlashInfer MLA: page_size = 1.
- FlashMLA: page_size = 64.
- Cutlass MLA: page_size = 128.
- TRTLLM MLA: page_size ∈ {32, 64}.

**中文对照**：MLA 页面大小约束：
- FlashInfer MLA：page_size = 1
- FlashMLA：page_size = 64
- Cutlass MLA：page_size = 128
- TRTLLM MLA：page_size ∈ {32, 64}

### Hybrid attention (different backends for prefill vs decode) (Experimental)

```{warning}
Hybrid attention is an experimental feature.
```

You can mix-and-match attention backends for prefill and decode. This is useful when one backend excels at prefill and another excels at decode. For the implementation details, please see `python/sglang/srt/layers/attention/hybrid_attn_backend.py`.

**中文对照**：您可以混合搭配预填充和解码的注意力后端。当一个后端在预填充方面表现出色，而另一个在解码方面表现出色时，这非常有用。关于实现细节，请参阅 `python/sglang/srt/layers/attention/hybrid_attn_backend.py`。

```bash
# Example: Prefill with FA4, Decode with TRTLLM MLA (Blackwell)
python3 -m sglang.launch_server \
  --model-path nvidia/DeepSeek-R1-FP4 \
  --tp 8 \
  --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm \
  --quantization modelopt_fp4 \
  --prefill-attention-backend fa4
```

#### Speculative decoding with hybrid attention

Hybrid attention also works with speculative decoding. The backend used for draft decoding and target verification depends on `--speculative-attention-mode`:

- `--speculative-attention-mode decode` (recommended): draft/verify use the decode backend.
- `--speculative-attention-mode prefill` (default): draft/verify use the prefill backend.

Constraints when combining hybrid attention with speculative decoding:

- If any attention backend is `trtllm_mha`, speculative decoding supports only `--speculative-eagle-topk 1`.
- For paged MHA backends with `--page-size > 1` and `--speculative-eagle-topk > 1`, only `flashinfer` is supported.
- CUDA Graph: the decode backend is always captured; the prefill backend is captured only when `--speculative-attention-mode prefill`.

**中文对照**：混合注意力也可以与推测解码一起使用。草稿解码和目标验证使用的后端取决于 `--speculative-attention-mode`：

- `--speculative-attention-mode decode`（推荐）：草稿/验证使用解码后端
- `--speculative-attention-mode prefill`（默认）：草稿/验证使用预填充后端

将混合注意力与推测解码结合时的约束：

- 如果任何注意力后端是 `trtllm_mha`，推测解码仅支持 `--speculative-eagle-topk 1`
- 对于 `--page-size > 1` 和 `--speculative-eagle-topk > 1` 的分页 MHA 后端，仅支持 `flashinfer`
- CUDA Graph：解码后端始终被捕获；预填充后端仅在 `--speculative-attention-mode prefill` 时被捕获


```{tip}
If you set only one of `--prefill-attention-backend` or `--decode-attention-backend`, the unspecified phase inherits `--attention-backend`.
If both are specified and differ, SGLang automatically enables a hybrid wrapper to dispatch to the chosen backend per phase.

**中文对照**：如果您只设置 `--prefill-attention-backend` 或 `--decode-attention-backend` 中的一个，未指定的阶段会继承 `--attention-backend`。如果两者都指定且不同，SGLang 会自动启用混合包装器以按阶段分派到选择的后端。
```

## Attention Backend Selection Guide (CUDA)

If the `--attention-backend` argument is not specified, SGLang automatically selects the best backend based on the hardware (CUDA) and model architecture.

**中文对照**：如果未指定 `--attention-backend` 参数，SGLang 会根据硬件（CUDA）和模型架构自动选择最佳后端。

### Automatic Selection Logic

**1. MHA Models (e.g., Llama, Qwen)**
- **Hopper (e.g., H100, H200)**: Defaults to `fa3` if using CUDA 12.3+ and the model configuration is supported.
- **Blackwell (e.g., B200)**: Defaults to `trtllm_mha`, unless using speculative decoding with `topk > 1`.
- **Other Architectures (Ampere, Ada, etc.)**: Defaults to `flashinfer` if available; otherwise falls back to `triton`.

**2. MLA Models (e.g., DeepSeek V3)**
- **Hopper**: Defaults to `fa3` (requires CUDA 12.3+).
- **Blackwell**: Defaults to `trtllm_mla`.
- **Other Architectures**: Defaults to `triton`.

**中文对照**：### 自动选择逻辑

**1. MHA 模型（例如 Llama、Qwen）**
- **Hopper（例如 H100、H200）**：如果使用 CUDA 12.3+ 且模型配置支持，默认为 `fa3`
- **Blackwell（例如 B200）**：默认为 `trtllm_mha`，除非使用 `topk > 1` 的推测解码
- **其他架构（Ampere、Ada 等）**：如果可用，默认为 `flashinfer`；否则回退到 `triton`

**2. MLA 模型（例如 DeepSeek V3）**
- **Hopper**：默认为 `fa3`（需要 CUDA 12.3+）
- **Blackwell**：默认为 `trtllm_mla`
- **其他架构**：默认为 `triton`


## User Guide

### Launch Command for Different Attention Backends

- FlashInfer (Default for Non-Hopper Machines, e.g., A100, A40)
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend flashinfer
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --attention-backend flashinfer \
  --trust-remote-code
```

- FlashAttention 3 (Default for Hopper Machines, e.g., H100, H200, H20)
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend fa3
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --trust-remote-code \
  --attention-backend fa3
```

- Triton
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend triton
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --attention-backend triton \
  --trust-remote-code
```

- FlashMLA
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend flashmla \
  --trust-remote-code
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend flashmla \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code
```

- TRTLLM MLA (Optimized for Blackwell Architecture, e.g., B200)
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend trtllm_mla \
  --trust-remote-code
```

- TRTLLM MLA with FP8 KV Cache (Higher concurrency, lower memory footprint)
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend trtllm_mla \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code
```

- FlashAttention 4 (MHA & MLA)
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --prefill-attention-backend fa4 \
  --trust-remote-code
```

- Cutlass MLA
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend cutlass_mla \
  --trust-remote-code
```

- Ascend
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend ascend
```

- Intel XPU
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend intel_xpu
```

- Wave
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend wave
```

- FlexAttention
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend flex_attention
```

- Dual Chunk FlashAttention
```bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-14B-Instruct-1M \
  --attention-backend dual_chunk_flash_attn
```

- Torch Native
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend torch_native
```

## 代码实现

### 核心文件
- `python/sglang/srt/layers/attention/`: 包含不同后端的具体实现：
    - `flashinfer_backend.py`: NVIDIA GPU 的高性能后端
    - `fa3_backend.py`: FlashAttention 3 实现
    - `triton_backend.py`: 可定制的基于 Triton 的实现
    - `base_attn_backend.py`: 定义抽象基类和接口

### 架构
注意力后端选择由 `python/sglang/srt/model_executor/model_runner.py` 协调。在初始化期间，`ModelRunner` 调用 `init_attention_backend()`，根据 `server_args.attention_backend` 或自动硬件检测（CUDA、ROCm、NPU 等）确定最优后端。它支持混合配置，即 prefill 和 decode 阶段通过 `HybridAttnBackend` 使用不同的后端。

### 关键代码逻辑
将字符串映射到后端类的选择逻辑：
```python
# model_runner.py
def _get_attention_backend_from_str(self, backend_str: str):
    backend_class = ATTENTION_BACKENDS[backend_str]
    return backend_class(self)
```
每个后端必须实现核心的 forward 方法：
```python
# base_attn_backend.py
def forward_extend(self, q, k, v, layer_id, forward_batch, ...):
    # Prefill 和 KV cache 扩展
def forward_decode(self, q, k, v, layer_id, forward_batch, ...):
    # 增量解码
```

### 集成要点
选定的 `self.attn_backend` 被集成到模型架构的 `RadixAttention` 层中。在前向传播期间，模型根据 `ForwardMode`（Prefill vs. Decode）将注意力计算分派到后端的 `forward_extend` 或 `forward_decode` 方法。

## Steps to add a new attention backend
To add a new attention backend, you can learn from the existing backends
(`python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`)
and follow the steps below.

1. Run without cuda graph. Support the two forward functions
    - forward_extend
        - Will be used for prefill, prefill with KV cache, and target verification
        - It will be called once per layer
    - forward_decode
        - Will be used for normal decode, and draft decode
        - It will be called once per layer
    - init_forward_metadata
        - Initialize the class and common metadata shared by all layers
        - Call the plan function for optimizations like split_kv
        - It will be called once per forward
2. Run with cuda graph. It has two phases (capture and replay) and you need to implement three functions
    - init_cuda_graph_state
        - It will be called once during life time
        - Create all common shared buffers
    - init_forward_metadata_capture_cuda_graph
        - It will be called before capturing a cuda graph
        - It is similar to init_forward_metadata but write the medatada to some pre-defined buffers
    - init_forward_metadata_replay_cuda_graph
        - It will be called before replaying a cuda graph
        - This function is in the critical path and needs to be fast
