# Deterministic Inference

## Why Deterministic Inference Matters

Deterministic inference ensures consistent LLM outputs across runs, which is critical for:
- **Reinforcement Learning**: Ensures consistent logprobs across runs, reducing stochastic noise and making RL training more stable, reproducible, and debuggable.
- **Testing & Debugging**: Enables reproducible validation
- **Production**: Improves reliability and user experience

Even with `temperature=0`, standard LLM inference can produce different outputs due to dynamic batching and varying reduction orders in GPU kernels.

**中文对照**：## 为什么确定性推理很重要

确定性推理确保 LLM 输出在运行之间保持一致，这对于：
- **强化学习**：确保跨运行的 logprobs 一致，减少随机噪声，使 RL 训练更稳定、可重现和可调试
- **测试和调试**：实现可重现的验证
- **生产**：提高可靠性和用户体验

即使使用 `temperature=0`，标准 LLM 推理也可能产生不同的输出，这是由于动态批处理和 GPU 内核中不同的归约顺序造成的。

## The Root Cause of Non-Determinism

The main source is **varying batch sizes**. Different batch sizes cause GPU kernels to split reduction operations differently, leading to different addition orders. Due to floating-point non-associativity (`(a + b) + c ≠ a + (b + c)`), this produces different results even for identical inputs.

**中文对照**：主要来源是**批量大小的变化**。不同的批量大小会导致 GPU 内核以不同的方式分割归约操作，导致不同的加法顺序。由于浮点数的非结合性（`(a + b) + c ≠ a + (b + c)`），即使对于相同的输入，这也会产生不同的结果。


## SGLang's Solution

Building on [Thinking Machines Lab's batch-invariant operators](https://github.com/thinking-machines-lab/batch_invariant_ops), SGLang achieves fully deterministic inference while maintaining compatibility with chunked prefill, CUDA graphs, radix cache, and non-greedy sampling. The development roadmap for deterministic inference features can be found in this [issue](https://github.com/sgl-project/sglang/issues/10278).

### Supported Backends

Deterministic inference is only supported with the following three attention backends: **FlashInfer**, **FlashAttention 3 (FA3)**, and **Triton**.

**中文对照**：基于 [Thinking Machines Lab 的批次不变性算子](https://github.com/thinking-machines-lab/batch_invariant_ops)，SGLang 实现完全确定性推理，同时保持与分块预填充、CUDA graph、radix cache 和非贪心采样的兼容性。确定性推理功能的开发路线图可以在这个 [issue](https://github.com/sgl-project/sglang/issues/10278) 中找到。

### Supported Backends

Deterministic inference is only supported with the following three attention backends: **FlashInfer**, **FlashAttention 3 (FA3)**, and **Triton**.

**中文对照**：### 支持的后端

确定性推理仅支持以下三种注意力后端：**FlashInfer**、**FlashAttention 3 (FA3)** 和 **Triton**。

The following table shows feature compatibility for deterministic inference across different attention backends:

| Attention Backend | CUDA Graph | Chunked Prefill | Radix Cache | Non-greedy Sampling (Temp > 0) |
|-------------------|------------|-----------------|-------------|---------------------|
| **FlashInfer** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **FlashAttention 3 (FA3)** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Triton** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |

## Usage

### Basic Usage

Enable deterministic inference by adding the `--enable-deterministic-inference` flag:

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

### Server Arguments

| Argument | Type/Default | Description |
|----------|--------------|-------------|
| `--enable-deterministic-inference` | flag; default: disabled | Enable deterministic inference with batch-invariant operations |
| `--attention-backend` | string; default: fa3 | Choose attention backend (flashinfer, fa3, or triton) |

**中文对照**：## 使用方法

### 基本用法

通过添加 `--enable-deterministic-inference` 标志来启用确定性推理：

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

### 服务器参数

| 参数 | 类型/默认值 | 描述 |
|------|------------|------|
| `--enable-deterministic-inference` | 标志；默认：禁用 | 使用批次不变性操作启用确定性推理 |
| `--attention-backend` | 字符串；默认：fa3 | 选择注意力后端（flashinfer、fa3 或 triton） |

### Example Configurations

#### Qwen3-8B
```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference
```

#### Llama Models
```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

#### Qwen3-30B-A3B (MoE Model)
```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

### Deterministic Inference with Non-Greedy Sampling (Temperature > 0)

SGLang supports deterministic inference even with non-greedy sampling by using sampling seeds. This is particularly useful for reinforcement learning scenarios like GRPO (Group Relative Policy Optimization) where you need multiple diverse but reproducible responses.

#### Default Behavior

By default, SGLang uses a sampling seed of `42` for reproducible sampling:

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Tell me a joke",
        "sampling_params": {
            "temperature": 0.8,  # Non-greedy sampling
            "max_new_tokens": 128,
        },
    },
)
print(response.json())
# This will always produce the same response across runs
```

**中文对照**：SGLang 即使在使用非贪心采样时也支持确定性推理，通过使用采样种子。这对于像 GRPO（组相对策略优化）这样的强化学习场景特别有用，您需要多个多样化但可重现的响应。

#### 默认行为

默认情况下，SGLang 使用采样种子 `42` 来进行可重现的采样：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Tell me a joke",
        "sampling_params": {
            "temperature": 0.8,  # Non-greedy sampling
            "max_new_tokens": 128,
        },
    },
)
print(response.json())
# This will always produce the same response across runs
```

#### Generating Multiple Reproducible Responses

To sample different responses from the same prompt while maintaining reproducibility (e.g., for GRPO training), provide different sampling seeds in your requests:

```python
import requests

# Prepare a list of sampling seeds for different responses
sampling_seeds = [42, 43, 44, 45, 46]

responses = []
for seed in sampling_seeds:
    response = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": "Tell me a joke",
            "sampling_params": {
                "temperature": 0.8,
                "max_new_tokens": 128,
                "sampling_seed": seed,  # Specify sampling seed
            },
        },
    )
    responses.append(response.json())

# Each seed will produce a different but reproducible response
# Using the same seed will always produce the same response
```

This approach ensures that:
- Different seeds produce diverse responses
- The same seed always produces the same response across different runs
- Results are reproducible for debugging and evaluation

**中文对照**：这种方法确保：
- 不同的种子产生不同的响应
- 相同的种子在不同运行中总是产生相同的响应
- 结果可重现用于调试和评估


## Verification

Run deterministic tests to verify consistent outputs:

```bash
# Single test: same prompt, varying batch sizes
python3 -m sglang.test.test_deterministic --test-mode single --n-trials 50

# Prefix test: prompts with different prefix lengths
python3 -m sglang.test.test_deterministic --test-mode prefix --n-trials 50

# Radix Cache Consistency mode: test radix cache determinism (cached vs uncached prefill)
python3 -m sglang.test.test_deterministic --test-mode radix_cache
```

Expected result: All tests should show `Unique samples: 1` (perfectly deterministic).

**中文对照**：运行确定性测试以验证一致的输出：

```bash
# 单次测试：相同的提示词，变化的批量大小
python3 -m sglang.test.test_deterministic --test-mode single --n-trials 50

# 前缀测试：不同前缀长度的提示词
python3 -m sglang.test.test_deterministic --test-mode prefix --n-trials 50

# Radix Cache 一致性模式：测试 radix cache 确定性（缓存 vs 非缓存预填充）
python3 -m sglang.test.test_deterministic --test-mode radix_cache
```

预期结果：所有测试应显示 `Unique samples: 1`（完全确定性）。

## 代码实现

### 核心文件

本功能主要由以下文件实现：

1. **[python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py](../../python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py)**
   - 主要职责：批次不变性算子的核心实现（基于 Triton 和 DeepGEMM）
   - 关键函数：
     - `enable_batch_invariant_mode()` - 启用批次不变性模式
     - `matmul_kernel_persistent()` - Triton JIT 编译的持久化矩阵乘法内核
     - `batch_invariant_linear()` - 批次不变的线性层前向传播
   - 技术细节：使用 persistent kernel 和 tile-based scheduling 确保计算顺序一致

2. **[python/sglang/srt/layers/layernorm.py](../../python/sglang/srt/layers/layernorm.py)**
   - 主要职责：批次不变的 LayerNorm 实现
   - 关键类：支持批次不变性的 RMS Norm 和 Layer Norm
   - 技术细节：固定 reduction 顺序，避免浮点数累加顺序的变化

3. **[python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py)**
   - 主要职责：在模型运行器中启用批次不变性模式
   - 关键代码位置：`model_runner.py:619-623`
   - 集成逻辑：检查 `server_args.enable_deterministic_inference` 标志，启动时调用 `enable_batch_invariant_mode()`

4. **[python/sglang/srt/layers/sampler.py](../../python/sglang/srt/layers/sampler.py)**
   - 主要职责：支持确定性采样（通过 sampling_seed）
   - 关键功能：即使 temperature > 0，使用相同 seed 也能产生相同结果

### 架构

确定性推理的实现架构：

```
启动时初始化
└─> model_runner.py::__init__()
    └─> if enable_deterministic_inference:
        └─> batch_invariant_ops.enable_batch_invariant_mode()
            └─> 全局开关：替换 torch.nn.functional.linear 等算子

推理时调用
├─> 线性层计算
│   └─> batch_invariant_linear() (替代 F.linear)
│       └─> matmul_kernel_persistent() [Triton kernel]
│           └─> 固定的 tile 遍历顺序（与 batch size 无关）
│
├─> LayerNorm 计算
│   └─> batch_invariant_layernorm() (替代标准 LayerNorm)
│       └─> 固定的 reduction 顺序
│
└─> Token 采样
    └─> sampler.py::forward()
        └─> torch.manual_seed(sampling_seed) (确定性随机)
            └─> 相同 seed → 相同采样结果
```

### 关键代码逻辑

**功能点 1：批次不变性模式启用**
- 实现位置：`batch_invariant_ops.py` - `enable_batch_invariant_mode()` 函数
- 核心逻辑：
  ```python
  # 全局上下文管理器，替换标准算子
  def enable_batch_invariant_mode():
      global _batch_invariant_mode_enabled
      _batch_invariant_mode_enabled = True
      # 替换 F.linear, F.layer_norm 等算子为批次不变版本
  ```

**功能点 2：Triton 持久化内核**
- 实现位置：`batch_invariant_ops.py:69-150` (`matmul_kernel_persistent()`)
- 核心逻辑：
  - 使用 `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)` 固定 tile 遍历顺序
  - Tile ID 计算与 batch size 无关，只依赖矩阵维度
  - 累加器使用固定的 `tl.dot(a, b, accumulator)` 顺序

**功能点 3：DeepGEMM 后端支持**
- 实现位置：`batch_invariant_ops.py:18-30`
- 核心逻辑：
  - 环境变量 `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=1` 启用 DeepGEMM
  - DeepGEMM 提供批次不变的 GEMM 实现（性能优于 Triton）

**功能点 4：注意力后端兼容性**
- 实现位置：`server_args.py` - `--attention-backend` 参数
- 支持的后端：
  - **FlashInfer**：完全支持（除 Radix Cache 外）
  - **FlashAttention 3 (FA3)**：完全支持（包括 Radix Cache）
  - **Triton**：完全支持（包括 Radix Cache）

**功能点 5：确定性采样**
- 实现位置：`sampler.py` - `forward()` 方法中的 seed 处理
- 核心逻辑：
  ```python
  if sampling_params.sampling_seed is not None:
      torch.manual_seed(sampling_params.sampling_seed)
  # 后续采样结果完全由 seed 决定
  ```

### 集成要点

**配置参数**：
- `server_args.py` 中的 `--enable-deterministic-inference` 标志
- `--attention-backend {flashinfer, fa3, triton}` - 选择支持确定性的后端

**启动流程**：
- `engine.py` → `model_runner.py` 初始化时检查标志
- 调用 `batch_invariant_ops.enable_batch_invariant_mode()` 启用全局模式

**运行时行为**：
- 所有线性层和归一化层自动使用批次不变算子
- 不需要修改模型代码，通过 monkey-patching 实现
- 相同输入 + 相同 batch size → 完全相同的 logits
- 不同 batch size → logits 数值略有不同，但对于相同输入，重复运行结果一致

**性能影响**：
- **Triton 后端**：约 5-10% 的性能开销（相比非确定性版本）
- **DeepGEMM 后端**：性能开销更小，接近原生性能
- **内存开销**：无额外内存开销

**环境变量**：
- `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=1`：启用 DeepGEMM（推荐）
- `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=1`：允许回退到非确定性 GEMM
- `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_COMPARISON_TEST=1`：启用正确性对比测试
