# DeepSeek V3.2 Usage

DeepSeek-V3.2 model family equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attention mechanism powered by a lightning indexer, DeepSeek-V3.2 achieves efficiency improvements in long-context scenarios.

**中文对照**：DeepSeek-V3.2 模型系列通过持续训练为 DeepSeek-V3.1-Terminus 配备了 DeepSeek 稀疏注意力（DSA）。借助由快速索引器驱动的细粒度稀疏注意力机制，DeepSeek-V3.2 在长上下文场景中实现了效率提升。

For reporting issues or tracking upcoming features, please refer to this [Roadmap](https://github.com/sgl-project/sglang/issues/11060).

**中文对照**：如需报告问题或跟踪即将推出的功能，请参阅此[路线图](https://github.com/sgl-project/sglang/issues/11060)。

Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) or [DeepSeek-V3.2-Speciale](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale) is the same as DeepSeek-V3.2-Exp except for the tool call parser.

**中文对照**：注意：本文档最初是为 [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) 模型的用法而编写的。[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) 或 [DeepSeek-V3.2-Speciale](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale) 的用法与 DeepSeek-V3.2-Exp 相同，只是工具调用解析器不同。

## Installation

**中文对照**：安装

### Docker

```bash
# H200/B200
docker pull lmsysorg/sglang:latest

# MI350/MI355
docker pull lmsysorg/sglang:v0.5.8-rocm700-mi35x

# MI300
# v0.5.8-rocm700-mi30x does not include PR #17504. Prefer the newest MI30x ROCm
# image tag from Docker Hub when available, or build from source (below).
docker pull lmsysorg/sglang:v0.5.8-rocm700-mi30x


# NPUs
docker pull lmsysorg/sglang:dsv32-a2
docker pull lmsysorg/sglang:dsv32-a3
```

### Build From Source

```bash
# Install SGLang
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install -e "python"
```
## Launch DeepSeek V3.2 with SGLang

To serve [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) on 8xH200/B200 GPUs:

```bash
# Launch with TP + DP (Recommended)
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention

# Launch with EP + DP
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep 8 --dp 8 --enable-dp-attention

# Launch with Pure TP
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8

# Launch with TP on MI30x/MI35x
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --nsa-prefill-backend tilelang --nsa-decode-backend tilelang
```

### Configuration Tips

**中文对照**：配置提示

- **DP Attention (Recommended)**: For DeepSeek V3.2 model, the kernels are customized for the use case of `dp_size=8`, so DP attention (`--dp 8 --enable-dp-attention`) is the recommended configuration for better stability and performance. All test cases use this configuration by default.

**中文对照**：**DP 注意力（推荐）**：对于 DeepSeek V3.2 模型，内核针对 `dp_size=8` 的用例进行了定制，因此 DP 注意力（`--dp 8 --enable-dp-attention`）是推荐的配置，可提供更好的稳定性和性能。所有测试用例默认使用此配置。

- **Pure TP Mode**: Launching with pure TP (without `--dp` and `--enable-dp-attention`) is also supported. Note that this mode has not been fully validated in PD disaggregation scenarios.

**中文对照**：**纯 TP 模式**：也支持使用纯 TP 启动（不使用 `--dp` 和 `--enable-dp-attention`）。请注意，此模式在 PD 分解场景中尚未完全验证。

- **Short-sequence MHA prefill (adaptive)**: For short prefill sequences (default threshold: **2048 tokens**), the NSA backend uses standard MHA automatically (no extra flags). On H200 (SM90) this path uses the FlashAttention variable-length kernel; on B200 (SM100) it uses TRT-LLM ragged MHA. MHA uses `MHA_ONE_SHOT` for best performance. `MHA_ONE_SHOT` computes multi-head attention over all tokens (both cached prefix and newly extended tokens) in a single kernel invocation, avoiding the overhead of chunked KV cache processing. This achieves optimal throughput for short sequences where total sequence length fits within the chunk capacity limit.

**中文对照**：**短序列 MHA 预填充（自适应）**：对于短预填充序列（默认阈值：**2048 个令牌**），NSA 后端自动使用标准 MHA（无需额外标志）。在 H200（SM90）上，此路径使用 FlashAttention 可变长度内核；在 B200（SM100）上，它使用 TRT-LLM ragged MHA。MHA 使用 `MHA_ONE_SHOT` 以获得最佳性能。`MHA_ONE_SHOT` 在单次内核调用中计算所有令牌的多头注意力（包括缓存的前缀和新扩展的令牌），避免了分块 KV 缓存处理的开销。这在总序列长度适合块容量限制的短序列中实现了最佳吞吐量。

- **Choices of Attention Kernels**: The attention backend is automatically set to `nsa` attention backend for DeepSeek V3.2 model. In this backend, different kernels for sparse prefilling/decoding are implemented, which can be specified by `--nsa-prefill-backend` and `--nsa-decode-backend` server arguments. The choices of nsa prefill/decode attention kernels include:

**中文对照**：**注意力内核选择**：对于 DeepSeek V3.2 模型，注意力后端自动设置为 `nsa` 注意力后端。在此后端中，实现了用于稀疏预填充/解码的不同内核，可以通过 `--nsa-prefill-backend` 和 `--nsa-decode-backend` 服务器参数来指定。nsa 预填充/解码注意力内核的选择包括：

  - `flashmla_sparse`: `flash_mla_sparse_fwd` kernel from `flash_mla` library. Can run on both Hopper and Blackwell GPUs. It requires bf16 q, kv inputs.

**中文对照**：- `flashmla_sparse`：来自 `flash_mla` 库的 `flash_mla_sparse_fwd` 内核。可以在 Hopper 和 Blackwell GPU 上运行。它需要 bf16 q、kv 输入。

  - `flashmla_kv`: `flash_mla_with_kvcache` kernel from `flash_mla` library. Can run on both Hopper and Blackwell GPUs. It requires bf16 q, fp8 k_cache inputs.

**中文对照**：- `flashmla_kv`：来自 `flash_mla` 库的 `flash_mla_with_kvcache` 内核。可以在 Hopper 和 Blackwell GPU 上运行。它需要 bf16 q、fp8 k_cache 输入。

  - `fa3`: `flash_attn_with_kvcache` kernel from `flash_attn` library. Can only run on Hopper GPUs. It requires bf16 q, kv inputs.

**中文对照**：- `fa3`：来自 `flash_attn` 库的 `flash_attn_with_kvcache` 内核。只能在 Hopper GPU 上运行。它需要 bf16 q、kv 输入。

  - `tilelang`: `tilelang` implementation that can run on GPU, HPU and NPU.

**中文对照**：- `tilelang`：可以在 GPU、HPU 和 NPU 上运行的 `tilelang` 实现。

  - `aiter`: Aiter kernel on AMD HPUs. Can only be used as decode kernel.

**中文对照**：- `aiter`：AMD HPU 上的 Aiter 内核。仅可用作解码内核。

- On the basis of performance benchmarks, the default configuration on H200 and B200 are set as follows :

**中文对照**：基于性能基准测试，H200 和 B200 上的默认配置设置如下：

  - H200: `flashmla_sparse` prefill attention (short-seq prefill uses MHA via FlashAttention varlen), `fa3` decode attention, `bf16` kv cache dtype.

**中文对照**：- H200：`flashmla_sparse` 预填充注意力（短序列预填充通过 FlashAttention varlen 使用 MHA）、`fa3` 解码注意力、`bf16` kv 缓存数据类型。

  - B200: `flashmla_auto` prefill attention (short-seq prefill uses MHA via TRT-LLM ragged), `flashmla_kv` decode attention, `fp8_e4m3` kv cache dtype. `flashmla_auto` enables automatic selection of either `flashmla_sparse` or `flashmla_kv` kernel for prefill based on KV cache dtype, hardware, and heuristics. When FP8 KV cache is enabled and `total_kv_tokens < total_q_tokens * 512`, it uses the `flashmla_sparse` kernel; otherwise, it falls back to the `flashmla_kv` kernel. The heuristics may need to be tuned if the performance of either the `flashmla_sparse` or `flashmla_kv` kernel changes significantly.

**中文对照**：- B200：`flashmla_auto` 预填充注意力（短序列预填充通过 TRT-LLM ragged 使用 MHA）、`flashmla_kv` 解码注意力、`fp8_e4m3` kv 缓存数据类型。`flashmla_auto` 根据 KV 缓存数据类型、硬件和启发式方法，自动选择 `flashmla_sparse` 或 `flashmla_kv` 内核进行预填充。当启用 FP8 KV 缓存且 `total_kv_tokens < total_q_tokens * 512` 时，它使用 `flashmla_sparse` 内核；否则回退到 `flashmla_kv` 内核。如果 `flashmla_sparse` 或 `flashmla_kv` 内核的性能发生显著变化，可能需要调整启发式方法。

## Multi-token Prediction

**中文对照**：多令牌预测

SGLang implements Multi-Token Prediction (MTP) for DeepSeek V3.2 based on [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding). With this optimization, the decoding speed can be improved significantly on small batch sizes. Please look at [this PR](https://github.com/sgl-project/sglang/pull/11652) for more information.

**中文对照**：SGLang 基于 [EAGLE 推测解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding) 为 DeepSeek V3.2 实现了多令牌预测（MTP）。使用此优化，可以在小批次大小下显著提高解码速度。请参阅[此 PR](https://github.com/sgl-project/sglang/pull/11652) 了解更多详情。

Example usage with DP Attention:

**中文对照**：DP 注意力的示例用法：
```bash
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

Example usage with Pure TP:

**中文对照**：纯 TP 的示例用法：
```bash
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

- The best configuration for `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` can be searched with [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) script for given batch size. The minimum configuration is `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`, which can achieve speedup for larger batch sizes.

**中文对照**：- 可以使用 [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) 脚本针对给定批次大小搜索 `--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 的最佳配置。最小配置为 `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`，可以实现较大批次大小的加速。

- The default value of  `--max-running-requests` is set to `48` for MTP. For larger batch sizes, this value should be increased beyond the default value.

**中文对照**：- `--max-running-requests` 的默认值对于 MTP 设置为 `48`。对于较大的批次大小，应将此值增加到默认值以上。

```{tip}
To enable the experimental overlap scheduler for EAGLE speculative decoding, set the environment variable `SGLANG_ENABLE_SPEC_V2=1`. This can improve performance by enabling overlap scheduling between draft and verification stages.
```


## Function Calling and Reasoning Parser

**中文对照**：函数调用和推理解析器

The usage of function calling and reasoning parser is the same as DeepSeek V3.1. Please refer to [Reasoning Parser](https://docs.sglang.io/advanced_features/separate_reasoning.html) and [Tool Parser](https://docs.sglang.io/advanced_features/tool_parser.html) documents.

**中文对照**：函数调用和推理解析器的用法与 DeepSeek V3.1 相同。请参阅[推理解析器](https://docs.sglang.io/advanced_features/separate_reasoning.html)和[工具解析器](https://docs.sglang.io/advanced_features/tool_parser.html)文档。

To launch `DeepSeek-V3.2-Exp` with function calling and reasoning parser:

**中文对照**：使用函数调用和推理解析器启动 `DeepSeek-V3.2-Exp`：

> Note: It is recommended to specify the chat-template, ensuring that you are within the sglang's root directory.

**中文对照**：注意：建议指定 chat-template，确保你位于 sglang 的根目录中。
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --tool-call-parser deepseekv31 \
  --reasoning-parser deepseek-v3 \
  --chat-template ./examples/chat_template/tool_chat_template_deepseekv32.jinja
```

To launch `DeepSeek-V3.2` with function calling and reasoning parser:

**中文对照**：使用函数调用和推理解析器启动 `DeepSeek-V3.2`：
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2 \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --tool-call-parser deepseekv32 \
  --reasoning-parser deepseek-v3
```

`DeepSeek-V3.2-Speciale` doesn't support tool calling, so can only be launched with reasoning parser:

**中文对照**：`DeepSeek-V3.2-Speciale` 不支持工具调用，因此只能使用推理解析器启动：
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Speciale \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --reasoning-parser deepseek-v3
```

## NVFP4 Checkpoint

**中文对照**：NVFP4 检查点

To launch deepseek v3.2 [NVFP4 checkpoint](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4) on Blackwell devices, the user needs to specify the quantization method as `modelopt_fp4`, and moe runner backend as one of `flashinfer_trtllm`(recommended), `flashinfer_cutlass` and `flashinfer_cutedsl`. Any other usage (parallelism, reasoning parser, ...) is the same as FP8 checkpoint.

**中文对照**：要在 Blackwell 设备上启动 deepseek v3.2 [NVFP4 检查点](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4)，用户需要将量化方法指定为 `modelopt_fp4`，并将 moe 运行时后端指定为 `flashinfer_trtllm`（推荐）、`flashinfer_cutlass` 或 `flashinfer_cutedsl` 中的一个。任何其他用法（并行性、推理解析器等）与 FP8 检查点相同。

An example launching command can be:

**中文对照**：启动命令示例：
```bash
python -m sglang.launch_server --model nvidia/DeepSeek-V3.2-NVFP4 --tp 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --tool-call-parser deepseekv32  --reasoning-parser deepseek-v3
```

## PD Disaggregation

Prefill Command:
```bash
python -m sglang.launch_server \
        --model-path deepseek-ai/DeepSeek-V3.2-Exp \
        --disaggregation-mode prefill \
        --host $LOCAL_IP \
        --port $PORT \
        --tp 8 \
        --dp 8 \
        --enable-dp-attention \
        --dist-init-addr ${HOST}:${DIST_PORT} \
        --trust-remote-code \
        --disaggregation-bootstrap-port 8998 \
        --mem-fraction-static 0.9 \
```

Decode command:
```bash
python -m sglang.launch_server \
        --model-path deepseek-ai/DeepSeek-V3.2-Exp \
        --disaggregation-mode decode \
        --host $LOCAL_IP \
        --port $PORT \
        --tp 8 \
        --dp 8 \
        --enable-dp-attention \
        --dist-init-addr ${HOST}:${DIST_PORT} \
        --trust-remote-code \
        --mem-fraction-static 0.9 \
```

Router command:
```bash
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill $PREFILL_ADDR 8998 \
  --decode $DECODE_ADDR \
  --host 127.0.0.1 \
  --port 8000 \
```

If you need more advanced deployment methods or production-ready deployment methods, such as RBG or LWS-based deployment, please refer to [references/multi_node_deployment/rbg_pd/deepseekv32_pd.md](../references/multi_node_deployment/rbg_pd/deepseekv32_pd.md). Additionally, you can also find startup commands for DeepEP-based EP parallelism in the aforementioned documentation.


## Benchmarking Results

**中文对照**：基准测试结果

### Accuracy Test with `gsm8k`

**中文对照**：使用 `gsm8k` 进行准确率测试

A simple accuracy benchmark can be tested with `gsm8k` dataset:

**中文对照**：可以使用 `gsm8k` 数据集进行简单的准确率基准测试：
```bash
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
```

The result is 0.956, which matches our expectation:

**中文对照**：结果为 0.956，符合我们的预期：
```bash
Accuracy: 0.956
Invalid: 0.000
Latency: 25.109 s
Output throughput: 5226.235 token/s
```

To test long-context accuracy, run gsm8k with `--num-shots 20`. The results are very close to the 8 shots results:

**中文对照**：要测试长上下文准确率，使用 `--num-shots 20` 运行 gsm8k。结果与 8 次shot的结果非常接近：
```
Accuracy: 0.956
Invalid: 0.000
Latency: 29.545 s
Output throughput: 4418.617 token/s
```


### Accuracy Test with `gpqa-diamond`

**中文对照**：使用 `gpqa-diamond` 进行准确率测试

Accuracy benchmark on long context can be tested on GPQA-diamond dataset with long output tokens and thinking enabled:

**中文对照**：可以在 GPQA-diamond 数据集上使用长输出令牌和启用思考功能进行长上下文准确率基准测试：
```bash
python3 -m sglang.test.run_eval --port 30000 --eval-name gpqa --num-examples 198 --max-tokens 128000 --repeat 8 --thinking-mode deepseek-v3
```

The mean accuracy over 8 runs shows 0.797, which matches the number 0.799 in official tech report.

**中文对照**：8 次运行的平均准确率为 0.797，与官方技术报告中的 0.799 相符。
```bash
Repeat: 8, mean: 0.797
Scores: ['0.808', '0.798', '0.808', '0.798', '0.783', '0.788', '0.803', '0.793']
```

For Deepseek V3.2, Deepseek recommends setting the sampling parameters to temperature = 1.0, top_p = 0.95:

**中文对照**：对于 Deepseek V3.2，Deepseek 建议将采样参数设置为 temperature = 1.0，top_p = 0.95：

```bash
python3 -m sglang.test.run_eval --port 30000 --eval-name gpqa --num-examples 198 --max-tokens 128000 --repeat 8 --top-p 0.95 --temperature 1.0 --thinking-mode deepseek-v3

Repeat: 8, mean: 0.840
Scores: ['0.848', '0.808', '0.848', '0.838', '0.879', '0.813', '0.838', '0.848']
```
which matches the official score, 0.824, as reported in the [Deepseek-V3.2 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf).

### Accuracy Test with `aime 2025`

**中文对照**：使用 `aime 2025` 进行准确率测试

Prepare the environment by installing NeMo-Skills in the docker or your own virtual environment:

**中文对照**：通过在 docker 或自己的虚拟环境中安装 NeMo-Skills 来准备环境：

  ```
  pip install git+https://github.com/NVIDIA/NeMo-Skills.git --ignore-installed blinker
  ```

Then launch the SGLang server:

**中文对照**：然后启动 SGLang 服务器：
```
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention
```

**For `DeepSeek-V3.2` and `DeepSeek-V3.2-Speciale`**:

```
python3 -m sglang.launch_server   --model-path deepseek-ai/DeepSeek-V3.2   --trust-remote-code   --tp-size 8 --dp-size 8 --enable-dp-attention   --tool-call-parser deepseekv32   --reasoning-parser deepseek-v3
```

Run the following script to evaluate AIME 2025:

**中文对照**：运行以下脚本来评估 AIME 2025：
```
#! /bin/bash
export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1

ns prepare_data aime25

PORT=30000
BACKEND=sglang
MODEL="deepseek-ai/DeepSeek-V3.2-Exp" # Should be changed to the model name
MODEL_NAME="dsv32-fp8"

echo "Starting AIME25 evaluation with model $MODEL on port $PORT using backend $BACKEND..."
ns eval \
  --benchmarks=aime25:4 \
  --server_type=$BACKEND \
  --model=$MODEL \
  --server_address=http://localhost:${PORT}/v1 \
  --output_dir=nemo_skills_aime25_${MODEL_NAME}_output_${BACKEND}_$(date +%Y%m%d_%H%M%S) \
  ++chat_template_kwargs.thinking=true \
  ++inference.temperature=1.0 \
  ++inference.top_p=0.95 \
  ++inference.tokens_to_generate=64000
  # ++inference.tokens_to_generate=120000 for Speciale model
```

Test results (8*B200):

**中文对照**：测试结果（8*B200）：

DeepSeek-V3.2-Exp：

| evaluation_mode    | num_entries | avg_tokens | gen_seconds | symbolic_correct      | no_answer |
|--------------------|-------------|------------|-------------|-----------------------|-----------|
| pass@1[avg-of-4]   | 30          | 15040      | 1673        | 87.50% ± 1.67%        | 0.00%     |
| majority@4         | 30          | 15040      | 1673        | 90.00%                | 0.00%     |
| pass@4             | 30          | 15040      | 1673        | 90.00%                | 0.00%     |


DeepSeek-V3.2:
| evaluation_mode    | num_entries | avg_tokens | gen_seconds | symbolic_correct      | no_answer |
|--------------------|-------------|------------|-------------|-----------------------|-----------|
| pass@1[avg-of-4]   | 30          | 13550      | 1632        | 92.50% ± 1.67%        | 0.00%     |
| majority@4         | 30          | 13550      | 1632        | 94.71%                | 0.00%     |
| pass@4             | 30          | 13550      | 1632        | 96.67%                | 0.00%     |


DeepSeek-V3.2-Speciale:
| evaluation_mode    | num_entries | avg_tokens | gen_seconds | symbolic_correct      | no_answer |
|--------------------|-------------|------------|-------------|-----------------------|-----------|
| pass@1[avg-of-4]   | 30          | 24155      | 3583        | 95.00% ± 1.92%        | 0.00%     |
| majority@4         | 30          | 24155      | 3583        | 95.83%                | 0.00%     |
| pass@4             | 30          | 24155      | 3583        | 100.00%               | 0.00%     |



## DSA long sequence context parallel optimization(experimental)

**中文对照**：DSA 长序列上下文并行优化（实验性）

**Note: This feature is only verified on Hopper machines**

**中文对照**：**注意**：此功能仅在 Hopper 机器上验证

For context parallel in DeepSeek V3.2 model, we provide two different modes of splitting tokens, which can be controlled with argument `--nsa-prefill-cp-mode`.

**中文对照**：对于 DeepSeek V3.2 模型中的上下文并行，我们提供了两种不同的令牌分割模式，可以通过参数 `--nsa-prefill-cp-mode` 控制。

### In sequence splitting (default setting)

**中文对照**：序列内分割（默认设置）

The first mode can be enabled by `--nsa-prefill-cp-mode in-seq-split`. This mode implements context parallel for DSA by splitting the sequence uniformly between context parallel ranks. At attention stage, each cp rank computes the indexer results of sharded sequence, and collects the whole kv cache through all gather operator.

**中文对照**：第一种模式可以通过 `--nsa-prefill-cp-mode in-seq-split` 启用。该模式通过在上下文并行秩之间均匀分割序列来实现 DSA 的上下文并行。在注意力阶段，每个 cp 秩计算分片序列的索引器结果，并通过 all gather 操作收集整个 kv 缓存。

The communication group for context parallel reuses the one for attention tp, thus `cp_size` equals `atten_tp_size = tp_size / dp_size`.

**中文对照**：上下文并行的通信组复用了注意力 tp 的通信组，因此 `cp_size` 等于 `atten_tp_size = tp_size / dp_size`。

Note that in sequence splitting mode has the following restrictions:

**中文对照**：请注意，序列内分割模式有以下限制：

- The batch size is restricted to 1 for prefill batches

**中文对照**：- 预填充批次的批次大小限制为 1

- Multi-node/PD disaggregation is still not supported

**中文对照**：- 仍不支持多节点/PD 分解

- `moe_dense_tp_size=1`, `kv_cache_dtype = "bf16"`, `moe_a2a_backend = "deepep"`

**中文对照**：- `moe_dense_tp_size=1`、`kv_cache_dtype = "bf16"`、`moe_a2a_backend = "deepep"`

- To ensure `cp_size > 1`, the passed in `tp_size` must be larger than `dp_size`

**中文对照**：- 为确保 `cp_size > 1`，传入的 `tp_size` 必须大于 `dp_size`

For more details, please refer to PR https://github.com/sgl-project/sglang/pull/12065.

**中文对照**：更多详情请参阅 PR https://github.com/sgl-project/sglang/pull/12065。

Example:

**中文对照**：示例：
```bash
# In-seq splitting mode launched with EP + DP
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp  --tp 8 --ep 8 --dp 2 --enable-dp-attention --enable-nsa-prefill-context-parallel --nsa-prefill-cp-mode in-seq-split --max-running-requests 32
```

### Round robin splitting

**中文对照**：轮询分割

This mode can be enabled by specifying the parameter `--nsa-prefill-cp-mode round-robin-split`, which distributes tokens across ranks based on `token_idx % cp_size`.

**中文对照**：该模式可以通过指定参数 `--nsa-prefill-cp-mode round-robin-split` 来启用，该模式根据 `token_idx % cp_size` 在秩之间分布令牌。

In this scenario, compared with the aforementioned method, it additionally supports the fused MoE backend (the fused MoE backend may deliver better performance than DeepEP in single-machine scenarios), FP8 KV-cache, and multi-batch prefill inference. But it cannot be enabled with dp attention together.

**中文对照**：与前述方法相比，此场景还支持 fused MoE 后端（在单机场景中 fused MoE 后端可能比 DeepEP 提供更好的性能）、FP8 KV 缓存和多批次预填充推理。但它不能与 dp 注意力一起启用。

For more details, please refer to PR https://github.com/sgl-project/sglang/pull/13959.

**中文对照**：更多详情请参阅 PR https://github.com/sgl-project/sglang/pull/13959。

Example usage:

**中文对照**：示例用法：
```bash
# Launch with FusedMoe + CP8
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp  --tp 8 --enable-nsa-prefill-context-parallel --nsa-prefill-cp-mode round-robin-split --max-running-requests 32
```

### Pipeline Parallel + Context Parallel (PP + CP)

**中文对照**：流水线并行 + 上下文并行（PP + CP）

This mode combines Pipeline Parallelism (PP) and Context Parallelism (CP) to scale across multiple nodes, which can achieve better throughput and Time To First Token (TTFT). Note that this method has only been tested on H20 96G.

**中文对照**：该模式结合了流水线并行（PP）和上下文并行（CP）以在多个节点上扩展，可以实现更好的吞吐量和首令牌时间（TTFT）。请注意，此方法仅在 H20 96G 上测试过。

#### Standard Usage

**中文对照**：标准用法

To launch with PP=2 and CP (via `round-robin-split` mode) on 2 nodes. This configuration uses the fused MoE kernel by default, which generally provides better performance.

**中文对照**：在 2 个节点上使用 PP=2 和 CP（通过 `round-robin-split` 模式）启动。默认情况下，此配置使用 fused MoE 内核，通常提供更好的性能。

For related development details, please refer to:

**中文对照**：相关开发详情请参阅：

- Fused MoE + CP support: [PR #13959](https://github.com/sgl-project/sglang/pull/13959)

**中文对照**：- Fused MoE + CP 支持：[PR #13959](https://github.com/sgl-project/sglang/pull/13959)

- PP + CP support: [Issue #15358](https://github.com/sgl-project/sglang/issues/15358) and [PR #16380](https://github.com/sgl-project/sglang/pull/16380)

**中文对照**：- PP + CP 支持：[Issue #15358](https://github.com/sgl-project/sglang/issues/15358) 和 [PR #16380](https://github.com/sgl-project/sglang/pull/16380)

Node 0:
```bash
export SGLANG_PP_LAYER_PARTITION=30,31
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --nnodes 2 --node-rank 0 \
  --dist-init-addr <HEAD_NODE_IP>:62001 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --trust-remote-code \
  --disable-radix-cache \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 8 \
  --page-size 64 \
  --watchdog-timeout 3600 \
  --host 0.0.0.0 --port 8000 \
  --tool-call-parser deepseekv32
```

Node 1:
```bash
export SGLANG_PP_LAYER_PARTITION=30,31
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --nnodes 2 --node-rank 1 \
  --dist-init-addr <HEAD_NODE_IP>:62001 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --trust-remote-code \
  --disable-radix-cache \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 8 \
  --page-size 64 \
  --watchdog-timeout 3600 \
  --host 0.0.0.0 --port 8000 \
  --tool-call-parser deepseekv32
```

#### PD Disaggregation with PP + CP

**中文对照**：PP + CP 的 PD 分解

If using PD (Prefill-Decode) Disaggregation, the Prefill nodes can be configured with PP + CP as follows.

**中文对照**：如果使用 PD（预填充-解码）分解，可以按以下方式配置 PP + CP 的预填充节点。

Prefill Node 0:
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --served-model-name deepseek-v32 \
  --nnodes 2 --node-rank 0 \
  --dist-init-addr <PREFILL_HEAD_IP>:20102 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split  \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
  --trust-remote-code \
  --disable-radix-cache \
  --max-running-requests 512 \
  --chunked-prefill-size 4096 \
  --context-length 131072 \
  --mem-fraction-static 0.9 \
  --page-size 64 \
  --enable-metrics \
  --collect-tokens-histogram \
  --tokenizer-worker-num 8 \
  --host 0.0.0.0 --port 30000
```

Prefill Node 1:
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --served-model-name deepseek-v32-prefill \
  --nnodes 2 --node-rank 1 \
  --dist-init-addr <PREFILL_HEAD_IP>:20102 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split  \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
  --trust-remote-code \
  --disable-radix-cache \
  --max-running-requests 512 \
  --chunked-prefill-size 4096 \
  --context-length 131072 \
  --mem-fraction-static 0.9 \
  --page-size 64 \
  --enable-metrics \
  --collect-tokens-histogram \
  --tokenizer-worker-num 8 \
  --host 0.0.0.0 --port 30000
```

For the Decode nodes, it is recommended to use the **EP mode**.

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/deepseek_v2.py` | DeepSeek V3.2 模型架构，支持 NSA（原生稀疏注意力） |
| `python/sglang/srt/layers/attention/nsa/` | NSA 注意力后端：稀疏索引器、FlashMLA sparse/KV 核、MHA 单次回退 |
| `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` | DeepSeek 稀疏注意力令牌选择的 Lightning 索引器 |
| `python/sglang/srt/managers/scheduler_dp_attn_mixin.py` | V3.2 的 DP 注意力协调（推荐：`--dp 8 --enable-dp-attention`） |
| `python/sglang/srt/speculative/eagle_worker.py` | V3.2 的 EAGLE MTP 推测解码 |

### 关键代码逻辑

- **NSA 后端自动选择**：V3.2 的注意力后端自动设为 `nsa`；可通过 `--nsa-prefill-backend` 和 `--nsa-decode-backend` 配置（选项：`flashmla_sparse`、`flashmla_kv`、`fa3`、`tilelang`、`aiter`）
- **短序列 MHA 回退**：对于 < 2048 令牌的预填充序列，NSA 自动使用标准 MHA（`MHA_ONE_SHOT` 模式）
- **上下文并行**：使用 `--enable-nsa-prefill-context-parallel` 配合 `--nsa-prefill-cp-mode in-seq-split|round-robin-split` 实现长上下文扩展
- **PD 解聚**：使用 `--disaggregation-mode prefill/decode` 配合 `sglang_router.launch_router --pd-disaggregation`

### 集成要点

- **服务器参数**：`--tp 8 --dp 8 --enable-dp-attention`、`--nsa-prefill-backend`、`--nsa-decode-backend`、`--enable-nsa-prefill-context-parallel`、`--nsa-prefill-cp-mode`
- **量化**：FP8（默认），NVFP4 通过 `--quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm`
- **PP+CP**：流水线并行结合上下文并行，用于多节点长上下文服务
