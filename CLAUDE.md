# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

SGLang is a high-performance serving framework for large language models (LLMs) and multimodal models. It delivers low-latency, high-throughput inference across diverse hardware (NVIDIA/AMD/Intel GPUs, TPUs, CPUs) with features like RadixAttention prefix caching, continuous batching, tensor/pipeline/data parallelism, and structured outputs.

**Key Innovation**: RadixAttention - automatic prefix caching using radix trees, achieving 5-10x speedups on workloads with repeated prompts.

**Scale**: Powers 400,000+ GPUs worldwide, generating trillions of tokens daily for companies like xAI, AMD, NVIDIA, Intel, LinkedIn, and Cursor.

---

## Architecture Overview

SGLang Runtime (SRT) follows a pipeline architecture with four core managers:

```
Client Request
    ↓
[TokenizerManager] ──→ Token IDs
    ↓
[Scheduler] ──→ Batches + Memory Management
    ↓
[ModelRunner] ──→ GPU Inference (Prefill/Decode)
    ↓
[DetokenizerManager] ──→ Text Stream
    ↓
Client Response
```

### Core Components (python/sglang/srt/)

- **managers/**: Core pipeline managers
  - `tokenizer_manager.py`: Async tokenization with RadixCache lookup
  - `scheduler.py`: Continuous batching scheduler with FCFS/LPM policies
  - `tp_worker.py`: Tensor-parallel model execution coordinator
  - `detokenizer_manager.py`: Incremental detokenization for streaming
  - `data_parallel_controller.py`: Multi-replica load balancing (DP)

- **mem_cache/**: Memory and cache systems
  - `radix_cache.py`: **Critical** - RadixAttention implementation (SGLang's killer feature)
  - `memory_pool.py`: Paged GPU memory allocator (similar to vLLM's PagedAttention)

- **model_executor/**: GPU inference execution
  - `model_runner.py`: Forward pass orchestration (prefill/decode/mixed batches)
  - `forward_batch_info.py`: Batch metadata for GPU kernels

- **layers/**: Neural network layers with custom kernels
  - `attention/`: FlashAttention/FlashInfer/Triton attention kernels
  - `sampler.py`: Token sampling from logits (temperature, top-p, top-k)

- **constrained/**: Structured output generation
  - `xgrammar_backend.py`: FSM-based grammar enforcement (JSON Schema, Regex, EBNF)

- **lora/**: LoRA adapter serving
  - `lora_manager.py`: S-LoRA multi-adapter batching

- **distributed/**: Parallelism infrastructure
  - `parallel_state.py`: TP/PP/DP/EP process group management

### Key Design Patterns

1. **Async Pipeline**: Each manager runs in a separate process, communicating via ZMQ
2. **Continuous Batching**: Dynamic batching with iteration-level scheduling
3. **Zero-Copy**: Shared memory (mmap) for cross-process tensor passing
4. **Mixed Precision**: FP16/BF16/FP8/INT4/AWQ/GPTQ quantization support

---

## Common Development Commands

### Running the Server

```bash
# Basic server launch (single GPU)
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 30000

# With tensor parallelism (4 GPUs)
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B \
  --tp-size 4 \
  --port 30000

# With data parallelism (2 replicas, 4 GPUs each)
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B \
  --tp-size 4 \
  --dp-size 2 \
  --port 30000
```

### Running Tests

SGLang uses **unittest** (not pytest by default).

```bash
# Run a single test file
cd test/srt
python3 test_srt_endpoint.py

# Run a specific test method
python3 test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run a test suite (per-commit tests)
python3 run_suite.py --suite stage-b-test-small-1-gpu

# Run nightly performance tests
python3 run_suite.py --suite nightly-1-gpu --nightly

# Run registered tests (new CI system)
cd test
python run_suite.py --hw cuda --suite stage-b-test-small-1-gpu
```

**Important**: Tests are run via `python3 test_file.py`, NOT `pytest test_file.py`.

### Building and Installing

```bash
# Development install (editable mode)
cd python
pip install -e .

# Install with dependencies
pip install -e ".[all]"

# Build wheel
python setup.py bdist_wheel
```

The `setup.py` automatically compiles protobuf files (`sglang/srt/grpc/sglang_scheduler.proto`) during build.

### Benchmarking

```bash
# Offline throughput benchmark
python -m sglang.bench_offline_throughput \
  --model meta-llama/Llama-3.1-8B \
  --dataset-name random \
  --num-prompts 1000

# Online serving benchmark (with server running)
python -m sglang.bench_serving \
  --backend sglang \
  --port 30000 \
  --dataset-name random \
  --num-prompts 1000 \
  --request-rate 10
```

---

## Code Navigation Guide

### Adding Support for a New Model

1. **Define model architecture**: `python/sglang/srt/models/<model_name>.py`
   - Inherit from `torch.nn.Module`
   - Implement `forward()` for prefill and decode
   - Register in `python/sglang/srt/models/__init__.py`

2. **Add config parser**: `python/sglang/srt/configs/<model_name>.py` (if special config needed)

3. **Add attention kernel**: Reuse existing FlashAttention or add custom in `layers/attention/`

4. **Test**: Add test in `test/srt/models/test_<model_name>_correctness.py`

### Implementing a New Scheduling Policy

1. Edit `python/sglang/srt/managers/schedule_policy.py`
2. Add new policy to `CacheAwarePolicy` or `CacheAgnosticPolicy` enum
3. Implement sorting logic in `SchedulePolicy.calc_priority()`
4. Example policies:
   - **FCFS**: First-come-first-serve (fairness)
   - **LPM**: Longest prefix match (maximize RadixCache hits)
   - **DFS-WEIGHT**: Depth-first search weighting (balance cache and latency)

### Understanding Request Flow (Critical for Debugging)

Trace a request through the pipeline:

1. **Entry**: `entrypoints/http_server.py` → `generate()` API handler
2. **Tokenization**: `managers/tokenizer_manager.py` → `handle_generate_request()`
3. **Cache lookup**: `mem_cache/radix_cache.py` → `match_prefix()` (finds cached KV)
4. **Scheduling**: `managers/scheduler.py` → `get_next_batch_to_run()`
5. **GPU forward**: `managers/tp_worker.py` → `forward_batch_generation()`
6. **Sampling**: `layers/sampler.py` → `forward()` (logits → token IDs)
7. **Detokenization**: `managers/detokenizer_manager.py` → `create_abort_task()`
8. **Streaming**: `entrypoints/http_server.py` → SSE stream to client

**Debug tip**: Enable profiling with `--enable-profiler` to see per-stage latency breakdown.

---

## Chinese Learning Documentation

SGLang includes comprehensive **Chinese learning materials** in `learning-guide/`:

- **00-welcome.md**: Quick start for beginners
- **01-overview.md**: System overview and design philosophy
- **02-architecture.md**: Four-component pipeline architecture
- **03-server-startup.md**: Server initialization process
- **04-request-journey.md**: End-to-end request tracing
- **05-tokenization.md**: Tokenizer manager internals
- **06-scheduling.md**: Scheduler and continuous batching
- **07-model-inference.md**: GPU inference and CUDA graphs
- **08-kv-cache.md**: KV cache and RadixAttention deep dive
- **09-sampling.md**: Token sampling strategies
- **10-glossary.md**: Technical terminology
- **11-advanced-features.md**: Structured output, function calling, multimodal, LoRA
- **12-production-deployment.md**: Monitoring, distributed deployment, performance tuning

**32 source files** have inline Chinese comments (mixed style: metaphors for concepts, technical details for implementation).

---

## Testing Best Practices

### Test File Naming

- Place tests in `test/srt/` (backend) or `test/lang/` (frontend)
- Nightly tests go in `test/srt/nightly/`
- Use descriptive names: `test_<feature>_<scenario>.py`

### Test Registration (New CI System)

```python
from sglang.test.ci.ci_register import register_cuda_ci

# Per-commit test on RTX 5090 (preferred for most 1-GPU tests)
register_cuda_ci(est_time=80, suite="stage-b-test-small-1-gpu")

# Per-commit test on H100 (for large models or SM120-incompatible tests)
register_cuda_ci(est_time=120, suite="stage-b-test-large-1-gpu")

# 2-GPU test
register_cuda_ci(est_time=200, suite="stage-b-test-large-2-gpu")

# Nightly-only test
register_cuda_ci(est_time=200, suite="nightly-1-gpu", nightly=True)
```

**When to use H100 (`stage-b-test-large-1-gpu`) instead of 5090:**
- FA3 attention backend (requires SM≤90, not SM120)
- FP8/MXFP4 quantization (not supported on Blackwell)
- Models >30B parameters or >32GB VRAM
- Tests with known 5090 failures

### Writing Efficient Tests

- **Reuse servers**: Launching a server is slow (~30s). Share one server across multiple test methods.
- **Use small models**: Prefer `Qwen/Qwen2.5-0.5B-Instruct` for functional tests, not 70B models.
- **Minimize GPU usage**: Avoid 8-GPU tests for simple scenarios. Use 1-2 GPUs when possible.
- **Keep tests focused**: One test method = one scenario. Use clear assertion messages.

Example structure:
```python
import unittest
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    popen_launch_server,
)

class TestMyFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.process = popen_launch_server(cls.model, port=30000)

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()

    def test_basic_generation(self):
        # Test code here
        response = ...
        self.assertIn("expected text", response)

if __name__ == "__main__":
    unittest.main()
```

---

## Critical Files to Understand

### Performance-Critical Paths

1. **radix_cache.py** (lines 1-500): RadixCache core - prefix matching, insertion, eviction
2. **scheduler.py** (lines 300-800): Batch scheduling loop - the heart of continuous batching
3. **model_runner.py** (lines 200-600): GPU forward pass - CUDA graph capture, batch execution
4. **attention/flashinfer_backend.py**: Attention kernel dispatch - prefill vs decode paths

### Configuration Files

- **server_args.py**: 200+ command-line arguments for server configuration
- **model_config.py**: Model architecture auto-detection from HuggingFace config.json
- **load_config.py**: Model weight loading (safetensors/pt/gguf/awq formats)

### Distributed Systems

- **data_parallel_controller.py**: DP replica routing (round-robin, least-load strategies)
- **scheduler_pp_mixin.py**: Pipeline parallelism (micro-batching, P2P communication)
- **scheduler_dp_attn_mixin.py**: DP attention coordination (MLP sync, batch alignment)

---

## Development Workflow Tips

### Local Development Setup

```bash
# 1. Clone and install
git clone https://github.com/sgl-project/sglang.git
cd sglang/python
pip install -e ".[all]"

# 2. Install pre-commit hooks (if available)
pre-commit install

# 3. Run a quick smoke test
python3 test/srt/test_srt_endpoint.py
```

### Before Submitting a PR

1. **Run relevant test suite**: `python3 test/run_suite.py --suite stage-b-test-small-1-gpu`
2. **Check syntax**: Ensure all modified Python files parse without errors
3. **Add tests**: New features require test coverage in `test/registered/`
4. **Update docs**: If changing APIs, update `learning-guide/` docs (Chinese) or docstrings

### Debugging Tips

- **Enable detailed logging**: `export SGLANG_LOGGING_LEVEL=DEBUG`
- **Profile performance**: Launch with `--enable-profiler`, then call `/profile` API endpoint
- **Inspect batches**: Set `SGLANG_LOG_FORWARD_ITERS=1` to log every forward iteration
- **Check GPU memory**: `watch -n 0.5 nvidia-smi` to monitor VRAM usage
- **Trace requests**: Each request has a unique `rid` - grep logs for it

### Common Pitfalls

1. **CUDA OOM**: Reduce `--max-total-tokens` (KV cache size) or batch size
2. **Slow RadixCache**: Ensure `--disable-radix-cache false` (it's enabled by default)
3. **Import errors**: Some modules require specific hardware (e.g., `flashinfer` needs CUDA)
4. **Test failures on 5090**: Use `stage-b-test-large-1-gpu` (H100) if test needs FA3 or FP8

---

## Additional Resources

- **Official Docs**: https://docs.sglang.io/
- **Blog Posts**: https://lmsys.org/blog/ (technical deep dives)
- **Slack Community**: https://slack.sglang.io/
- **Weekly Dev Meeting**: https://meet.sglang.io/
- **Contribution Guide**: https://docs.sglang.io/developer_guide/contribution_guide.html

---

## Notes for AI Assistants

- **Code Style**: Follow PEP 8. Use type hints for public APIs.
- **Comments**: Add Chinese comments for core concepts (metaphors + technical details) when working in annotated files.
- **Performance**: SGLang prioritizes throughput and latency. Avoid unnecessary allocations, prefer in-place ops.
- **Compatibility**: Support NVIDIA, AMD, and CPU backends. Use `is_hip()`, `is_npu()` checks when needed.
- **Testing**: Always add unit tests for new features. Reuse existing server instances to speed up tests.
