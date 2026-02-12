# TPU

SGLang supports high-performance TPU inference through the SGLang-JAX backend, which is specifically optimized for Google Cloud TPUs. The JAX-based implementation delivers exceptional throughput and low latency for Large Language Model (LLM) serving workloads on TPU hardware.

For TPU-specific issues or feature requests, please visit the [sglang-jax GitHub issues page](https://github.com/sgl-project/sglang-jax/issues).

**NOTE:** SGLang TPU support is implemented via the SGLang-JAX backend, a dedicated JAX-based inference engine maintained as a separate repository at [https://github.com/sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax).

**中文对照**：# TPU

SGLang 通过 SGLang-JAX 后端支持高性能 TPU 推理，该后端针对 Google Cloud TPU 进行了专门优化。基于 JAX 的实现为 TPU 硬件上的大型语言模型（LLM）服务工作负载提供了卓越的吞吐量和低延迟。

对于 TPU 特定问题或功能请求，请访问 [sglang-jax GitHub issues 页面](https://github.com/sgl-project/sglang-jax/issues)。

**注意：** SGLang TPU 支持通过 SGLang-JAX 后端实现，这是一个专用的基于 JAX 的推理引擎，作为独立仓库在 [https://github.com/sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax) 维护。

## System Requirements

### Supported TPU Hardware

| TPU Type | HBM Memory | Availability |
|----------|-----------|--------------|
| TPU v6e | 32 GB | Google Cloud |
| TPU v7 | 96 GB per core | Google Cloud |

### Software Requirements

- **Python:** 3.12 or higher
- **JAX:** Latest version with TPU support
- **Environment:** Google Cloud TPU VM or compatible TPU runtime
- **Optional:** SkyPilot for simplified cloud deployment

**中文对照**：## 系统要求

### 支持的 TPU 硬件

| TPU 类型 | HBM 内存 | 可用性 |
|----------|-----------|--------------|
| TPU v6e | 32 GB | Google Cloud |
| TPU v7 | 每核心 96 GB | Google Cloud |

### 软件要求

- **Python:** 3.12 或更高版本
- **JAX:** 支持 TPU 的最新版本
- **环境:** Google Cloud TPU VM 或兼容的 TPU 运行时
- **可选:** SkyPilot 用于简化的云部署

## Feature Support Matrix

SGLang-JAX provides comprehensive TPU-optimized features for production LLM serving:

| Feature | Support Status | Description |
|---------|---------------|-------------|
| High-Throughput Continuous Batching | ✅ | Dynamic request batching for maximum TPU utilization |
| Radix Tree KV Cache | ✅ | Memory-efficient prefix sharing between requests |
| FlashAttention Backend | ✅ | TPU-optimized attention kernel for long sequences |
| Tensor Parallelism | ✅ | Distribute models across multiple TPU cores |
| Paged Attention | ✅ | Flexible KV cache management with paging |
| Speculative Decoding (EAGLE/EAGLE3) | ✅ | 20-40% throughput improvement for compatible models |
| Chunked Prefill | ✅ | Mixed prefill-decode batching |
| OpenAI-Compatible API | ✅ | Drop-in replacement for OpenAI API |
| Data Parallel Attention | 🚧 | In development - Attention computation with data parallelism |
| Quantization | 🚧 | In development - Model quantization for reduced memory usage |
| Multi-LoRA | 🚧 | In development - Serve multiple LoRA adapters simultaneously |

### Attention Backend Comparison

| Backend | Paged Attention | Spec Decoding | MLA | Sliding Window |
|---------|----------------|---------------|-----|----------------|
| FlashAttention (fa) | ✅ | ✅ | ❌ | ✅ |
| Native | ❌ | ❌ | ❌ | ❌ |

**NOTE:** FlashAttention backend is recommended for production workloads due to superior memory efficiency and performance.

**中文对照**：## 功能支持矩阵

SGLang-JAX 为生产级 LLM 服务提供全面的 TPU 优化功能：

| 功能 | 支持状态 | 描述 |
|---------|---------------|-------------|
| 高吞吐量连续批处理 | ✅ | 动态请求批处理以最大化 TPU 利用率 |
| Radix Tree KV 缓存 | ✅ | 请求之间内存高效的前缀共享 |
| FlashAttention 后端 | ✅ | 针对长序列优化的 TPU 注意力内核 |
| 张量并行 | ✅ | 跨多个 TPU 核心分布模型 |
| 分页注意力 | ✅ | 使用分页的灵活 KV 缓存管理 |
| 推测解码 (EAGLE/EAGLE3) | ✅ | 兼容模型吞吐量提升 20-40% |
| 块状预填充 | ✅ | 混合预填充-解码批处理 |
| OpenAI 兼容 API | ✅ | OpenAI API 的直接替代品 |
| 数据并行注意力 | 🚧 | 开发中 - 带数据并行的注意力计算 |
| 量化 | 🚧 | 开发中 - 用于减少内存使用的模型量化 |
| 多-LoRA | 🚧 | 开发中 - 同时服务多个 LoRA 适配器 |

### 注意力后端比较

| 后端 | 分页注意力 | 推测解码 | MLA | 滑动窗口 |
|---------|----------------|---------------|-----|----------------|
| FlashAttention (fa) | ✅ | ✅ | ❌ | ✅ |
| Native | ❌ | ❌ | ❌ | ❌ |

**注意：** 由于卓越的内存效率和性能，建议生产工作负载使用 FlashAttention 后端。

## Optimized Model List

The following models have been tested and optimized for TPU deployment:

| Model Family | Performance Status |
|--------------|-------------------|
| [Qwen 3](https://huggingface.co/Qwen) | ⭐ Recommended for production |
| [Qwen 3 MoE](https://huggingface.co/Qwen) | ⭐ Best performance |
| [Qwen 2](https://huggingface.co/Qwen) | Needs improvement |
| [Qwen 2 MoE](https://huggingface.co/Qwen) | Needs improvement |
| [Qwen 1.5](https://huggingface.co/Qwen) | Needs improvement |
| [Llama/LLaMA](https://huggingface.co/meta-llama) | Needs improvement |
| [Grok-2](https://huggingface.co/xai-org) | Needs improvement |
| [Gemma 2](https://huggingface.co/google) | Verified on TPU |
| Bailing MoE | Needs improvement |

**中文对照**：## 优化模型列表

以下模型已针对 TPU 部署进行了测试和优化：

| 模型系列 | 性能状态 |
|--------------|-------------------|
| [Qwen 3](https://huggingface.co/Qwen) | ⭐ 推荐用于生产 |
| [Qwen 3 MoE](https://huggingface.co/Qwen) | ⭐ 最佳性能 |
| [Qwen 2](https://huggingface.co/Qwen) | 需要改进 |
| [Qwen 2 MoE](https://huggingface.co/Qwen) | 需要改进 |
| [Qwen 1.5](https://huggingface.co/Qwen) | 需要改进 |
| [Llama/LLaMA](https://huggingface.co/meta-llama) | 需要改进 |
| [Grok-2](https://huggingface.co/xai-org) | 需要改进 |
| [Gemma 2](https://huggingface.co/google) | 已在 TPU 上验证 |
| Bailing MoE | 需要改进 |

## Installation

### Method 1: Using PyPI (Recommended)

```bash
pip install sglang-jax
```

### Method 2: From Source

```bash
git clone https://github.com/sgl-project/sglang-jax
cd sglang-jax
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e "python[all]"
```

### Method 3: Using Docker

**NOTE:** Docker support for TPU is currently under development. Please use PyPI or source installation methods.

### Method 4: Cloud TPU with SkyPilot

[SkyPilot](https://github.com/skypilot-org/skypilot) provides simplified deployment on Google Cloud TPU:

1. Install SkyPilot and configure GCP access (see [SkyPilot documentation](https://skypilot.readthedocs.io/))

2. Create a SkyPilot configuration file:

<details>
<summary>SkyPilot YAML: <code>sglang-jax.sky.yaml</code></summary>

```yaml
# sglang-jax.sky.yaml
resources:
   accelerators: tpu-v6e-4
   accelerator_args:
      tpu_vm: True
      runtime_version: v2-alpha-tpuv6e

run: |
  git clone https://github.com/sgl-project/sglang-jax.git
  cd sglang-jax
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -e "python[all]"
```

</details>

3. Launch your TPU cluster:

```bash
# Standard deployment
sky launch -c sglang-jax sglang-jax.sky.yaml --infra=gcp

# With spot instances for cost savings
sky launch -c sglang-jax sglang-jax.sky.yaml --infra=gcp --use-spot
```

**中文对照**：## 安装

### 方法 1：使用 PyPI（推荐）

```bash
pip install sglang-jax
```

### 方法 2：从源码安装

```bash
git clone https://github.com/sgl-project/sglang-jax
cd sglang-jax
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e "python[all]"
```

### 方法 3：使用 Docker

**注意**：TPU 的 Docker 支持目前正在开发中。请使用 PyPI 或源码安装方法。

### 方法 4：使用 SkyPilot 的云 TPU

[SkyPilot](https://github.com/skypilot-org/skypilot) 在 Google Cloud TPU 上提供简化的部署：

1. 安装 SkyPilot 并配置 GCP 访问（请参阅 [SkyPilot 文档](https://skypilot.readthedocs.io/)）

2. 创建 SkyPilot 配置文件：

<details>
<summary>SkyPilot YAML: <code>sglang-jax.sky.yaml</code></summary>

```yaml
# sglang-jax.sky.yaml
resources:
   accelerators: tpu-v6e-4
   accelerator_args:
      tpu_vm: True
      runtime_version: v2-alpha-tpuv6e

run: |
  git clone https://github.com/sgl-project/sglang-jax.git
  cd sglang-jax
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -e "python[all]"
```

</details>

3. 启动您的 TPU 集群：

```bash
# 标准部署
sky launch -c sglang-jax sglang-jax.sky.yaml --infra=gcp

# 使用竞价实例以节省成本
sky launch -c sglang-jax sglang-jax.sky.yaml --infra=gcp --use-spot
```

## Launch of the Serving Engine

### Basic Example: Qwen-7B

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen-7B-Chat \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=4 \
    --device=tpu \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=8192 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 30000
```

**Key Parameters Explained:**

1. `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` - Enables JIT compilation caching to accelerate server startup on subsequent runs
2. `--tp-size=4` - Tensor parallelism size; match this to your TPU core count (typically 1, 4, or 8)
3. `--device=tpu` - Specifies TPU device (this is the default for sglang-jax)
4. `--dtype=bfloat16` - Uses bfloat16 precision, which TPUs are optimized for
5. `--mem-fraction-static=0.8` - Allocates 80% of TPU HBM for static memory (adjustable from 0.2 to 0.9)
6. `--max-prefill-tokens=8192` - Maximum number of tokens processed in the prefill phase

**中文对照**：## 启动服务引擎

### 基本示例：Qwen-7B

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen-7B-Chat \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=4 \
    --device=tpu \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=8192 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 30000
```

**关键参数说明：**

1. `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` - 启用 JIT 编译缓存以加速后续运行的服务器启动
2. `--tp-size=4` - 张量并行大小；将其与您的 TPU 核心数匹配（通常为 1、4 或 8）
3. `--device=tpu` - 指定 TPU 设备（这是 sglang-jax 的默认值）
4. `--dtype=bfloat16` - 使用 bfloat16 精度，TPU 为此进行了专门优化
5. `--mem-fraction-static=0.8` - 为静态内存分配 80% 的 TPU HBM（可从 0.2 调整到 0.9）
6. `--max-prefill-tokens=8192` - 预填充阶段处理的最大令牌数

### High-Performance Configuration: Qwen3-8B

For production workloads with optimal throughput:

```bash
python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-8B \
    --trust-remote-code \
    --tp-size=4 \
    --device=tpu \
    --mem-fraction-static=0.8 \
    --chunked-prefill-size=2048 \
    --dtype=bfloat16 \
    --max-running-requests=256 \
    --page-size=128 \
    --attention-backend=fa
```

### Advanced: Speculative Decoding (EAGLE3)

Speculative decoding can improve throughput by 20-40% for compatible models:

```bash
python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-32B \
    --trust-remote-code \
    --device=tpu \
    --tp-size=4 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=4096 \
    --attention-backend=fa \
    --dtype=bfloat16 \
    --port=30000 \
    --host=0.0.0.0 \
    --disable-overlap-schedule \
    --speculative-algorithm=EAGLE3 \
    --speculative-draft-model-path=AngelSlim/Qwen3-32B_eagle3 \
    --page-size=64 \
    --speculative-eagle-topk=1 \
    --speculative-num-steps=3 \
    --speculative-num-draft-tokens=4
```

**NOTE:** Speculative decoding is currently supported for Qwen3 and LLaMA model families. See the [Speculative Decoding documentation](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md) for detailed configuration guidance.


### Multi-Node Distributed Serving

For large models requiring multiple TPU VMs:

```bash
# Node 0 (coordinator)
python3 -m sgl_jax.launch_server \
    --model-path MODEL_PATH \
    --dist-init-addr=NODE0_IP:10011 \
    --nnodes=2 \
    --node-rank=0 \
    --tp-size=8 \
    [other parameters...]

# Node 1 (worker)
python3 -m sgl_jax.launch_server \
    --model-path MODEL_PATH \
    --dist-init-addr=NODE0_IP:10011 \
    --nnodes=2 \
    --node-rank=1 \
    --tp-size=8 \
    [other parameters...]
```

## Benchmarking with Requests

### Throughput Testing

Basic throughput benchmark:

```bash
python3 -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --dataset-name random \
    --num-prompts=100 \
    --random-input=512 \
    --random-output=128 \
    --max-concurrency=8 \
    --random-range-ratio=1 \
    --warmup-requests=0
```

### Latency Testing

Measure single-batch latency:

```bash
python3 -m sgl_jax.bench_one_batch_server \
    --base-url http://127.0.0.1:30000 \
    --model-path Qwen/Qwen-7B-Chat \
    --batch-size=32 \
    --input-len=256 \
    --output-len=32
```

### Comprehensive Benchmark Script

For systematic performance evaluation across different configurations:

```bash
#!/bin/bash
set -e

backend=${1:-sgl-jax}
num_prompts_per_concurrency=3
input_seq_lens=(1024 4096 8192)
output_seq_lens=(1 1024)
max_concurrencies=(8 16 32 64 128 256)

for input_seq_len in "${input_seq_lens[@]}"; do
    for output_seq_len in "${output_seq_lens[@]}"; do
        echo "======================================="
        echo "Testing ISL/OSL: $input_seq_len/$output_seq_len"
        echo "======================================="
        for max_concurrency in "${max_concurrencies[@]}"; do
            num_prompts=$((num_prompts_per_concurrency * max_concurrency))
            python3 -m sgl_jax.bench_serving \
                --backend ${backend} \
                --dataset-name random \
                --num-prompts ${num_prompts} \
                --random-input ${input_seq_len} \
                --random-output ${output_seq_len} \
                --max-concurrency ${max_concurrency} \
                --random-range-ratio 1 \
                --disable-ignore-eos \
                --warmup-requests 0
        done
    done
done
```

For detailed help on all benchmark parameters:

```bash
python3 -m sgl_jax.bench_serving --help
```

See the [Benchmark and Profiling Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md) for advanced benchmarking techniques and profiling with JAX Profiler.

**中文对照**：## 使用请求进行基准测试

### 吞吐量测试

基本吞吐量基准测试：

```bash
python3 -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --dataset-name random \
    --num-prompts=100 \
    --random-input=512 \
    --random-output=128 \
    --max-concurrency=8 \
    --random-range-ratio=1 \
    --warmup-requests=0
```

### 延迟测试

测量单批次延迟：

```bash
python3 -m sgl_jax.bench_one_batch_server \
    --base-url http://127.0.0.1:30000 \
    --model-path Qwen/Qwen-7B-Chat \
    --batch-size=32 \
    --input-len=256 \
    --output-len=32
```

### 综合基准测试脚本

用于对不同配置进行系统的性能评估：

```bash
#!/bin/bash
set -e

backend=${1:-sgl-jax}
num_prompts_per_concurrency=3
input_seq_lens=(1024 4096 8192)
output_seq_lens=(1 1024)
max_concurrencies=(8 16 32 64 128 256)

for input_seq_len in "${input_seq_lens[@]}"; do
    for output_seq_len in "${output_seq_lens[@]}"; do
        echo "======================================="
        echo "Testing ISL/OSL: $input_seq_len/$output_seq_len"
        echo "======================================="
        for max_concurrency in "${max_concurrencies[@]}"; do
            num_prompts=$((num_prompts_per_concurrency * max_concurrency))
            python3 -m sgl_jax.bench_serving \
                --backend ${backend} \
                --dataset-name random \
                --num-prompts ${num_prompts} \
                --random-input ${input_seq_len} \
                --random-output ${output_seq_len} \
                --max-concurrency ${max_concurrency} \
                --random-range-ratio 1 \
                --disable-ignore-eos \
                --warmup-requests 0
        done
    done
done
```

有关所有基准测试参数的详细帮助：

```bash
python3 -m sgl_jax.bench_serving --help
```

请参阅[基准测试和分析指南](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md)了解高级基准测试技术和 JAX Profiler 分析。

## Performance Optimization

### Memory Optimization

**Reduce memory usage:**
- Lower `--mem-fraction-static` (from 0.8 → 0.5 → 0.3)
- Decrease `--max-prefill-tokens` (from 16384 → 8192 → 4096)
- Reduce `--max-running-requests`

**Handle OOM errors:**
- Start with conservative memory settings (`--mem-fraction-static=0.5`)
- Gradually increase until you find the optimal balance
- Increase `--page-size` for better memory locality (1 → 16 → 64 → 128)

### Throughput Optimization

To maximize tokens per second:

- Use FlashAttention backend: `--attention-backend=fa`
- Enable speculative decoding (EAGLE3) for Qwen3 models (20-40% improvement)
- Increase `--max-running-requests` to 256+
- Set `--mem-fraction-static` to 0.8+ (if memory allows)
- Use larger page sizes (64-128)
- Enable chunked prefill: `--chunked-prefill-size=2048`

### Latency Optimization

To minimize time-to-first-token (TTFT) and inter-token latency:

- Reduce `--page-size` to 1-4
- Lower `--max-running-requests` (16-32) for smaller batches
- Reduce `--chunked-prefill-size`
- Use conservative memory settings to avoid GC pauses

### TPU-Specific Optimizations

1. **JIT Compilation Cache:**
   ```bash
   export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
   ```
   Always set this environment variable to cache compiled kernels and accelerate server startup.

2. **Data Type Optimization:**
   Use `--dtype=bfloat16` for TPU native optimization. TPUs are specifically designed for bfloat16 computations.

3. **Tensor Parallelism:**
   Match `--tp-size` to your TPU core configuration (1, 4, or 8) for optimal model distribution.

4. **Attention Backend:**
   Always use `--attention-backend=fa` (FlashAttention) for production workloads.

## Troubleshooting

### OOM (Out of Memory) Errors

If you encounter out-of-memory errors:

1. Reduce `--mem-fraction-static` from 0.8 to 0.5 or lower
2. Decrease `--max-prefill-tokens` from 8192 to 4096 or 2048
3. Lower `--max-running-requests` to reduce concurrent batch size
4. Increase `--page-size` for better memory layout efficiency

### Compilation Long-Time

If the server takes too long to start:

1. Ensure `JAX_COMPILATION_CACHE_DIR` is properly set
2. Understand that the first run requires JIT compilation (this is normal)
3. Subsequent runs will be significantly faster with cached compilations
4. Consider using `--skip-server-warmup` to defer compilation until first request

### Low Throughput

If you're not achieving expected throughput:

1. Verify `--tp-size` matches your TPU core configuration
2. Check that `--attention-backend=fa` is enabled
3. Increase `--max-running-requests` to enable larger batch formation
4. Consider enabling speculative decoding for compatible models
5. Ensure memory settings allow for sufficient batch sizes

### Connection Issues

If clients cannot connect to the server:

1. Ensure `--host=0.0.0.0` for external access (not just `127.0.0.1`)
2. Verify firewall rules allow traffic on the specified port (default: 30000)
3. Check that the server process is running: `curl http://localhost:30000/health`

## Advanced Features

### Speculative Decoding

SGLang-JAX supports EAGLE and EAGLE3 speculative decoding algorithms for Qwen3 and LLaMA model families. Speculative decoding can improve throughput by 20-40% without affecting output quality.

See the [Speculative Decoding documentation](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md) for detailed configuration and supported model combinations.

### Chunked Prefill

Enable mixed prefill-decode batching for better TPU utilization:

```bash
--chunked-prefill-size=2048 --enable-mixed-chunk
```

This allows the scheduler to mix prefill operations with decode operations in the same batch, improving overall throughput.

### Custom Attention Backends

SGLang-JAX supports a plugin-based attention backend system. You can implement custom attention kernels optimized for specific use cases.

See the [Attention Backend documentation](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/attention_backend.md) for implementation details.

### Environment Verification

Verify your TPU setup before deploying:

```bash
python -c "from sgl_jax import check_env; check_env.check_env()"
```

This command checks:
- Installed package versions
- TPU device availability and specifications
- System resources and configuration
- Compatibility of settings

## Contributing

We welcome contributions to improve TPU support in SGLang-JAX!

### Areas for Contribution

**Check the [Development Roadmap](https://github.com/sgl-project/sglang-jax/issues/190)** to see planned features and find opportunities to contribute new functionality.

Current contribution areas include:

- Performance optimizations for specific TPU generations
- Support for additional model architectures
- Documentation improvements and examples
- Bug reports and fixes
- Benchmark results and performance analysis

### How to Contribute

1. Visit the [sglang-jax repository](https://github.com/sgl-project/sglang-jax)
2. Read the [Contribution Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/contribution_guide.md)
3. Join the [SGL-JAX Slack community](https://sgl-fru7574.slack.com/archives/C09EBE5HT5X) for discussions
4. Report issues at [sglang-jax/issues](https://github.com/sgl-project/sglang-jax/issues)

### Testing on TPU

For contributors who need TPU access for testing:

- Refer to the [TPU Resources Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/tpu_resources_guide.md) for information on accessing TPU hardware
- Use SkyPilot with spot instances for cost-effective testing
- Follow the [Benchmark and Profiling Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md) for performance validation

## References

### Documentation

- [SGLang-JAX Repository](https://github.com/sgl-project/sglang-jax)
- [SGLang-JAX Installation Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/get_started/install.md)
- [Qwen Models Quick Start](https://github.com/sgl-project/sglang-jax/blob/main/docs/basic_usage/qwen.md)
- [Benchmark and Profiling Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md)
- [Speculative Decoding](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md)

### External Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [SkyPilot Documentation](https://skypilot.readthedocs.io/)

## 代码实现

### 核心文件

SGLang TPU 支持通过独立的 [sglang-jax](https://github.com/sgl-project/sglang-jax) 仓库实现。

| 文件 | 作用 |
|------|------|
| `sgl_jax/launch_server.py` | TPU 服务器启动器：基于 JAX 的推理引擎入口点 |
| `sgl_jax/bench_serving.py` | TPU 基准测试脚本：吞吐量和延迟测量 |
| `sgl_jax/bench_one_batch_server.py` | TPU 的单批次延迟基准测试 |

### 关键代码逻辑

- **JAX 后端**：完全独立于主 CUDA/PyTorch 代码库；使用 JAX 进行 TPU 原生计算
- **JIT 编译缓存**：`JAX_COMPILATION_CACHE_DIR` 缓存编译的 TPU 内核，以加快后续启动
- **TPU 上的 FlashAttention**：通过 `--attention-backend=fa` 使用自定义 TPU 优化注意力内核
- **推测解码**：支持 Qwen3 和 LLaMA 系列的 EAGLE/EAGLE3（吞吐量提升 20-40%）

### 集成要点

- **安装**：`pip install sglang-jax`（独立 PyPI 包）或从 `sgl-project/sglang-jax` 源码构建
- **服务器启动**：`python3 -m sgl_jax.launch_server`（注意：`sgl_jax` 模块，而非 `sglang`）
- **基准测试**：`python3 -m sgl_jax.bench_serving --backend sgl-jax`（专用后端名称）
- **TPU 并行**：`--tp-size` 匹配 TPU 核心数（1、4 或 8）；通过 `--nnodes` 和 `--node-rank` 多节点
