# Benchmark and Profiling

## Benchmark

- Benchmark the latency of running a single static batch without a server. The arguments are the same as for `launch_server.py`.
  Note that this is a simplified test script without a dynamic batching server, so it may run out of memory for a batch size that a real server can handle. A real server truncates the prefill into several batches, while this simplified script does not.

**中文对照**：对在不启动服务器的情况下运行单个静态批次的延迟进行基准测试。参数与 `launch_server.py` 相同。请注意，这是一个没有动态批处理服务器的简化测试脚本，因此对于实际服务器可以处理的批次大小，它可能会耗尽内存。实际服务器会将预填充截断为多个批次，而此简化脚本不会。

  - Without a server (do not need to launch a server)
    ```bash
    python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch 32 --input-len 256 --output-len 32
    ```
  - With a server (please use `sglang.launch_server` to launch a server first and run the following command.)

**中文对照**：- 带服务器（请先使用 `sglang.launch_server` 启动服务器，然后运行以下命令。）

    ```bash
    python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch-size 32 --input-len 256 --output-len 32
    ```


- Benchmark offline processing. This script will start an offline engine and run the benchmark.

  ```bash
  python3 -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --num-prompts 10
  ```

**中文对照**：- 对离线处理进行基准测试。此脚本将启动离线引擎并运行基准测试。

- Benchmark online serving. Please use `sglang.launch_server` to launch a server first and run the following command.

  ```bash
  python3 -m sglang.bench_serving --backend sglang --num-prompt 10
  ```

**中文对照**：- 对在线服务进行基准测试。请先使用 `sglang.launch_server` 启动服务器，然后运行以下命令。

## Profile with PyTorch Profiler

[Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) is a convenient basic tool to inspect kernel execution time, call stack, and kernel overlap and occupancy.

**中文对照**：## 使用 PyTorch Profiler 进行分析

[PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 是一个方便的基本工具，用于检查内核执行时间、调用堆栈以及内核重叠和占用率。

### Profile a server with `sglang.bench_serving`

**中文对照**：### 使用 `sglang.bench_serving` 分析服务器

```bash
# set trace path
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# start server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# send profiling request from client
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile
```

The `SGLANG_TORCH_PROFILER_DIR` environment variable must be set on both the server and client side; otherwise, the trace file will not be generated correctly. A secure way to do this is by setting it in your shell's resource file (e.g., `~/.bashrc` for bash).

**中文对照**：`SGLANG_TORCH_PROFILER_DIR` 环境变量必须在服务器端和客户端都设置；否则，跟踪文件将无法正确生成。一种安全的方法是在 shell 的资源文件中设置它（例如，bash 的 `~/.bashrc`）。

For more details, please refer to [Bench Serving Guide](./bench_serving.md).

**中文对照**：更多详情，请参阅 [Bench Serving Guide](./bench_serving.md)。

### Profile In PD Disaggregation Mode

**中文对照**：### 在 PD 分离模式下进行分析

When profiling in PD disaggregation mode, prefill and decode workers **must be profiled separately** due to torch profiler limitations. The `bench_serving` command provides dedicated options for this:

**中文对照**：在 PD 分离模式下进行分析时，由于 torch 分析器的限制，预填充和解码工作进程**必须分别进行分析**。`bench_serving` 命令为此提供了专用选项：

#### Profile Prefill Workers

**中文对照**：#### 分析预填充工作进程

```bash
# set trace path
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# start prefill and decode servers (see PD disaggregation docs for setup)
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode prefill
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode decode --port 30001 --base-gpu-id 1

# start router
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000

# send profiling request targeting prefill workers
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile --pd-separated --profile-prefill-url http://127.0.0.1:30000
```

#### Profile Decode Workers

```bash
# send profiling request targeting decode workers
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile --pd-separated --profile-decode-url http://127.0.0.1:30001
```

#### Important Notes

- `--profile-prefill-url` and `--profile-decode-url` are **mutually exclusive** - you cannot profile both at the same time
- Both options support multiple worker URLs for multi-instance setups:
  ```bash
  # Profile multiple prefill workers
  python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --profile --pd-separated --profile-prefill-url http://127.0.0.1:30000 http://127.0.0.1:30002

  # Profile multiple decode workers
  python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --profile --pd-separated --profile-decode-url http://127.0.0.1:30001 http://127.0.0.1:30003
  ```
- Make sure `SGLANG_TORCH_PROFILER_DIR` is set on all worker nodes before starting the servers
- For more details on setting up PD disaggregation, see [PD Disaggregation Guide](../advanced_features/pd_disaggregation.md)

**中文对照**：#### 重要说明

- `--profile-prefill-url` 和 `--profile-decode-url` 是**互斥的**——你不能同时分析两者
- 两个选项都支持多实例设置中的多个工作进程 URL：
  - 分析多个预填充工作进程
  - 分析多个解码工作进程
- 确保在启动服务器之前在所有工作节点上设置 `SGLANG_TORCH_PROFILER_DIR`
- 有关设置 PD 分离的更多详情，请参阅 [PD Disaggregation Guide](../advanced_features/pd_disaggregation.md)

### Profile a server with `sglang.bench_offline_throughput`

**中文对照**：### 使用 `sglang.bench_offline_throughput` 分析服务器
```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# profile one batch with bench_one_batch.py
# batch size can be controlled with --batch argument
python3 -m sglang.bench_one_batch --model-path meta-llama/Llama-3.1-8B-Instruct --batch 32 --input-len 1024 --output-len 10 --profile

# profile multiple batches with bench_offline_throughput.py
python -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompts 10 --profile --mem-frac=0.8
```

### Profile a server with `sglang.profiler`

**中文对照**：### 使用 `sglang.profiler` 分析服务器

When the server is running (e.g., processing a decoding request), you can start live profiling immediately by sending a profile request to the server.

**中文对照**：当服务器正在运行时（例如，正在处理解码请求），你可以通过发送分析请求立即启动实时分析。

You can do this by running `python3 -m sglang.profiler`. For example:

```
# Terminal 1: Send a generation request
python3 -m sglang.test.send_one

# Terminal 2: Before the above request finishes, quickly launch the following command in a separate terminal.
# It will generate a profile of the above request for several decoding batches.
python3 -m sglang.profiler
```

**中文对照**：你可以通过运行 `python3 -m sglang.profiler` 来做到这一点。例如：

You can also combine the above operations into a single command

```
python3 -m sglang.test.send_one --profile
```

**中文对照**：你也可以将上述操作合并为一个命令

### Profile a server with HTTP API endpoints

**中文对照**：### 使用 HTTP API 端点分析服务器

SGLang provides HTTP API endpoints to control profiling on a running server. This allows you to start and stop profiling programmatically, which is useful for capturing specific workload patterns.

**中文对照**：SGLang 提供了 HTTP API 端点来控制运行中的服务器上的分析。这允许你以编程方式启动和停止分析，这对于捕获特定工作负载模式非常有用。

#### Using `/start_profile` endpoint

**中文对照**：#### 使用 `/start_profile` 端点

The `/start_profile` endpoint starts profiling on the server. You can control when profiling begins and how long it runs using the following parameters:

**Basic usage:**

```bash
# Start profiling immediately for 10 steps
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "num_steps": 10
  }'
```

**Parameters:**

- `output_dir` (optional): Directory where profile traces will be saved. If not specified, uses `SGLANG_TORCH_PROFILER_DIR` environment variable, or `/tmp` as the default
- `num_steps` (optional): Number of steps to profile. If not specified, profiling continues until manually stopped with `/end_profile`
- `start_step` (optional): Step number at which to start profiling (inclusive). Useful for skipping warmup iterations
- `activities` (optional): List of activities to profile, e.g., `["CPU", "GPU"]`. Default is `["CPU", "GPU"]`
- `merge_profiles` (optional): Whether to merge distributed traces. Default is `false`

**中文对照**：**参数：**
- `output_dir`（可选）：保存分析跟踪的目录。如果未指定，使用 `SGLANG_TORCH_PROFILER_DIR` 环境变量，或默认为 `/tmp`
- `num_steps`（可选）：要分析的步数。如果未指定，分析将继续直到手动使用 `/end_profile` 停止
- `start_step`（可选）：开始分析的步骤编号（含）。用于跳过预热迭代
- `activities`（可选）：要分析的活动列表，例如 `["CPU", "GPU"]`。默认为 `["CPU", "GPU"]`
- `merge_profiles`（可选）：是否合并分布式跟踪。默认为 `false`

**Note on step ranges:** Profiling starts at `start_step` (inclusive) and continues for `num_steps` iterations. For example, with `start_step=3` and `num_steps=10`, profiling captures steps 3, 4, 5, 6, 7, 8, 9, 10, 11, and 12 (10 steps total, starting from step 3).

**中文对照**：**关于步骤范围的说明：**分析从 `start_step`（含）开始，持续 `num_steps` 次迭代。例如，当 `start_step=3` 和 `num_steps=10` 时，分析捕获步骤 3、4、5、6、7、8、9、10、11 和 12（共 10 步，从步骤 3 开始）。

**Advanced usage with `start_step`:**

**中文对照**：**带有 `start_step` 的高级用法：**

```bash
# Wait 5 steps (warmup), then profile for 10 steps
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/profiles",
    "start_step": 5,
    "num_steps": 10,
    "activities": ["CPU", "GPU"]
  }'
```

**Continuous profiling (manual stop):**

```bash
# Start profiling without num_steps - must manually stop with /end_profile
curl -X POST http://127.0.0.1:30000/start_profile
```

**中文对照**：**持续分析（手动停止）：**

#### Using `/end_profile` endpoint

**中文对照**：#### 使用 `/end_profile` 端点

The `/end_profile` endpoint stops an ongoing profiling session and saves the trace file.

```bash
# Stop profiling and save traces
curl -X POST http://127.0.0.1:30000/end_profile
```

**中文对照**：`/end_profile` 端点停止正在进行的分析会话并保存跟踪文件。

This is only needed when you start profiling without specifying `num_steps`. If `num_steps` is specified, profiling will automatically stop after that many steps.

**中文对照**：仅当你在未指定 `num_steps` 的情况下启动分析时才需要此操作。如果指定了 `num_steps`，分析将在达到该步数后自动停止。

#### Example workflow

**中文对照**：#### 示例工作流程

```bash
# Terminal 1: Start the server
export SGLANG_TORCH_PROFILER_DIR=/tmp/profiles
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# Terminal 2: Start continuous profiling
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "start_step": 3
  }'

# Terminal 3: Send requests to generate load
python -m sglang.bench_serving --backend sglang --num-prompts 100

# Terminal 2: Stop profiling when done
curl -X POST http://127.0.0.1:30000/end_profile
```

### Profiler Trace Merger for Distributed Traces

SGLang now supports automatic merging of profiling traces from distributed setups with multiple parallelism types (TP, DP, PP, EP). This feature is particularly useful for analyzing performance across distributed runs.

**中文对照**：### 用于分布式跟踪的分析器跟踪合并器

SGLang 现在支持自动合并来自具有多种并行类型（TP、DP、PP、EP）的分布式设置的分析跟踪。此功能对于分析分布式运行中的性能特别有用。

#### Multi-Node Profiling and Shared Storage Considerations

**中文对照**：#### 多节点分析和共享存储注意事项

Single-node profiler output merging is completely supported. When profiling in distributed environments spanning multiple nodes, shared storage (e.g., NFS, Lustre) should be accessible by all nodes for the output directory to enable merging of trace files.

**中文对照**：完全支持单节点分析器输出合并。当在跨多个节点的分布式环境中进行分析时，共享存储（例如 NFS、Lustre）应可供所有节点访问输出目录，以启用跟踪文件合并。

If there is no shared storage accessible across nodes, automatic merging of trace files during profiling is not supported directly as of now.

**中文对照**：如果没有跨节点可访问的共享存储，则目前不支持在分析期间直接自动合并跟踪文件。

#### HTTP API Usage

**中文对照**：#### HTTP API 用法

```bash
# Start profiling with automatic trace merging enabled
curl -X POST <BASE_URL>/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/profiles", # where to store profile traces
    "num_steps": 10,
    "activities": ["CPU", "GPU"],
    "merge_profiles": true # optional argument to merge profile traces (default=False)
  }'
```

**中文对照**：`"merge_profiles": true # 可选参数，用于合并分析跟踪（默认=False）`

#### Command Line Usage

```bash
# Start profiling with merge enabled
python -m sglang.profiler \
  --num-steps 10 \
  --cpu \
  --gpu \
  --output-dir /tmp/profiles \
  --merge-profiles # optional argument to merge profile traces (default=False)
```

**中文对照**：#### 命令行用法

#### Output Files

**中文对照**：#### 输出文件

The profile merger generates:
- Individual rank trace files: `{profile_id}-TP-{tp}-DP-{dp}-PP-{pp}-EP-{ep}.trace.json.gz`
- Merged trace file: `merged-{profile_id}.trace.json.gz`

**中文对照**：分析合并器生成：
- 单独的 rank 跟踪文件：`{profile_id}-TP-{tp}-DP-{dp}-PP-{pp}-EP-{ep}.trace.json.gz`
- 合并的跟踪文件：`merged-{profile_id}.trace.json.gz`

### Possible PyTorch bugs

**中文对照**：### 可能的 PyTorch 错误
If in any cases you encounter the following error (for example, using qwen 2.5 VL):
```bash
RuntimeError: !stack.empty() INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/autograd/profiler_python.cpp":983, please report a bug to PyTorch. Python replay stack is empty.
```
This is likely a PyTorch Bug reported in [Bug: vLLM Profiler](https://github.com/vllm-project/vllm/issues/18240) and [Bug: torch.profiler.profile](https://github.com/pytorch/pytorch/issues/101632). As a workaround, you may disable `with_stack` with an environment variable such as follows:
```bash
export SGLANG_PROFILE_WITH_STACK=False
python -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompts 10 --profile --mem-frac=0.8
```

### View traces

Trace files can be loaded and visualized from:

1. https://ui.perfetto.dev/ (any browser)
2. chrome://tracing (Chrome browser only)

**中文对照**：### 查看跟踪

跟踪文件可以从以下位置加载和可视化：

1. https://ui.perfetto.dev/（任何浏览器）
2. chrome://tracing（仅限 Chrome 浏览器）

If browser cannot open trace file due to its large size,
client can generate a small trace file (<100MB) by controlling number of prompts and lengths of prompt outputs.
For example, when profiling a server,

```bash
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 2 --sharegpt-output-len 100 --profile
```

**中文对照**：如果浏览器由于文件太大而无法打开跟踪文件，客户端可以通过控制提示词数量和输出长度来生成小的跟踪文件（<100MB）。例如，当分析服务器时，

This command sets the number of prompts to 2 with `--num-prompts` argument and limits the length of output sequences to 100 with `--sharegpt-output-len` argument, which can generate a small trace file for browser to open smoothly.

**中文对照**：此命令使用 `--num-prompts` 参数将提示词数量设置为 2，并使用 `--sharegpt-output-len` 参数将输出序列长度限制为 100，这可以生成一个小的跟踪文件，以便浏览器顺畅打开。

Additionally, if you want to locate the SGLang Python source code through the cuda kernel in Trace, you need to disable CUDA Graph when starting the service. This can be done by using the `--disable-cuda-graph` parameter in the command to start the service.

**中文对照**：此外，如果你想通过跟踪中的 cuda 内核定位 SGLang Python 源代码，则需要在启动服务时禁用 CUDA Graph。这可以通过在启动服务的命令中使用 `--disable-cuda-graph` 参数来完成。

## Profile with Nsight

[Nsight systems](https://docs.nvidia.com/nsight-systems/) is an advanced tool that exposes more profiling details, such as register and shared memory usage, annotated code regions and low-level CUDA APIs and events.

**中文对照**：## 使用 Nsight 进行分析

[Nsight systems](https://docs.nvidia.com/nsight-systems/) 是一个高级工具，可显示更多分析细节，例如寄存器和共享内存使用情况、注释的代码区域以及低级 CUDA API 和事件。

1. Prerequisite:

   Install using apt, or run inside a [NVIDIA Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) or [SGLang Docker container](https://github.com/sgl-project/sglang/tree/main/docker).

   ```bash
   # install nsys
   # https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html
   apt update
   apt install -y --no-install-recommends gnupg
   echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
   apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   apt update
   apt install nsight-systems-cli
   ```

   **中文对照**：1. 先决条件：

   使用 apt 安装，或在 [NVIDIA Docker 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) 或 [SGLang Docker 容器](https://github.com/sgl-project/sglang/tree/main/docker) 中运行。

2. To profile a single batch, use

   ```bash
   nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch --model meta-llama/Meta-Llama-3-8B --batch-size 64 --input-len 512
   ```

   **中文对照**：2. 要分析单个批次，使用

3. To profile a server, e.g.

   ```bash
   # launch the server, set the delay and duration times according to needs
   # after the duration time has been used up, server will be killed by nsys

   nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o sglang.out --delay 60 --duration 70 python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache

   # client
   python3 -m sglang.bench_serving --backend sglang --num-prompts 1000 --dataset-name random --random-input 1024 --random-output 512
   ```

   In practice, we recommend users to set `--duration` argument to a large value. Whenever user wants the server to stop profiling. Firstly run:

   ```bash
   nsys sessions list
   ```

   to get the session id in the form of `profile-XXXXX`, then run:

   ```bash
   nsys stop --session=profile-XXXXX
   ```

   to manually kill the profiler and generate `nsys-rep` files instantly.

   **中文对照**：3. 要分析服务器，例如

   在实践中，我们建议用户将 `--duration` 参数设置为一个较大的值。每当用户想要服务器停止分析时。首先运行：

   以获取形式为 `profile-XXXXX` 的会话 ID，然后运行：

   以手动终止分析器并立即生成 `nsys-rep` 文件。

4. Use NVTX to annotate code regions, e.g. to see their execution time.

   ```bash
   # install nvtx
   pip install nvtx
   ```

   ```python
   # code snippets
   import nvtx
   with nvtx.annotate("description", color="color"):
       # some critical code
   ```

   **中文对照**：4. 使用 NVTX 注释代码区域，例如查看它们的执行时间。

### Layer-wise NVTX Profiling with Nsight Systems

**中文对照**：### 使用 Nsight Systems 进行逐层 NVTX 分析

SGLang provides built-in layerwise NVTX annotations that can be combined with the CUDA Profiler for detailed per-layer profiling in Nsight Systems. This is particularly useful for identifying performance bottlenecks at the layer level.

**中文对照**：SGLang 提供了内置的逐层 NVTX 注释，可以与 CUDA Profiler 结合使用，在 Nsight Systems 中进行详细的逐层分析。这对于识别层级的性能瓶颈特别有用。

#### Using `--enable-layerwise-nvtx-marker` with Nsight Systems and `/start_profile`

The `--enable-layerwise-nvtx-marker` flag automatically adds NVTX markers to every layer in your model. This is particularly powerful when combined with Nsight Systems profiling to see detailed per-layer performance.

**中文对照**：#### 将 `--enable-layerwise-nvtx-marker` 与 Nsight Systems 和 `/start_profile` 结合使用

`--enable-layerwise-nvtx-marker` 标志会自动为模型中的每一层添加 NVTX 标记。当与 Nsight Systems 分析结合使用时，可以查看详细的每层性能，特别强大。

**Method 1: Using `/start_profile` with CUDA_PROFILER (for programmatic control)**

This method allows you to control exactly when profiling starts/stops via HTTP API while Nsight Systems is running.

**中文对照**：**方法 1：使用 `/start_profile` 和 CUDA_PROFILER（用于程序化控制）**

此方法允许你在 Nsight Systems 运行时通过 HTTP API 精确控制分析的开始/停止时间。

1. Launch the server with layerwise NVTX enabled under Nsight Systems:

   ```bash
   # Terminal 1: Start server with nsys and capture-range option
   nsys profile --trace-fork-before-exec=true \
     --cuda-graph-trace=node \
     --capture-range=cudaProfilerApi \
     --capture-range-end=stop \
     -o layerwise_profile \
     python -m sglang.launch_server \
       --model-path meta-llama/Llama-3.1-8B-Instruct \
       --enable-layerwise-nvtx-marker \
       --disable-cuda-graph
   ```

   Note: NVTX markers are not emitted for kernel launches captured by CUDA graphs. Use `--disable-cuda-graph` to ensure all layerwise NVTX markers are emitted in the trace.

   **中文对照**：1. 在 Nsight Systems 下启用逐层 NVTX 启动服务器：

   注意：对于由 CUDA graph 捕获的内核启动，不会发出 NVTX 标记。使用 `--disable-cuda-graph` 确保所有逐层 NVTX 标记都在跟踪中发出。

2. In another terminal, control profiling via `/start_profile` with `CUDA_PROFILER` activity:

   ```bash
   # Terminal 2: Wait for server to be ready, then start CUDA profiling
   # Wait 3 steps for warmup, then profile for 10 steps
   curl -X POST http://127.0.0.1:30000/start_profile \
     -H "Content-Type: application/json" \
     -d '{
       "start_step": 3,
       "num_steps": 10,
       "activities": ["CUDA_PROFILER"]
     }'
   ```

   **中文对照**：2. 在另一个终端中，通过 `/start_profile` 和 `CUDA_PROFILER` 活动控制分析：

3. Send requests to generate load:

   ```bash
   # Terminal 3: Generate workload
   python -m sglang.bench_serving --backend sglang --num-prompts 100
   ```

   **中文对照**：3. 发送请求以生成负载：

4. Profiling will automatically stop after 10 steps (due to `num_steps: 10`). If you hadn't specified `num_steps`, you would need to manually stop it:

   ```bash
   # Terminal 2: Only needed if num_steps was not specified
   curl -X POST http://127.0.0.1:30000/end_profile
   ```

   **中文对照**：4. 分析将在 10 步后自动停止（由于 `num_steps: 10`）。如果你没有指定 `num_steps`，则需要手动停止它：

The `--capture-range=cudaProfilerApi` option tells Nsight Systems to only capture data between `cudaProfilerStart()` and `cudaProfilerStop()` calls (triggered by `/start_profile` and `/end_profile`), reducing overhead and file size. The `start_step` parameter skips the first 3 steps to avoid capturing warmup overhead.

**中文对照**：`--capture-range=cudaProfilerApi` 选项告诉 Nsight Systems 仅捕获 `cudaProfilerStart()` 和 `cudaProfilerStop()` 调用之间的数据（由 `/start_profile` 和 `/end_profile` 触发），减少开销和文件大小。`start_step` 参数跳过前 3 步以避免捕获预热开销。

**Method 2: Simpler approach without `/start_profile` API**

For simpler use cases where you don't need fine-grained control over profiling start/stop, you can profile with Nsight Systems capturing the entire workload:

```bash
# Terminal 1: Start server with layerwise NVTX
# Note: --disable-cuda-graph ensures all NVTX markers are emitted
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --enable-layerwise-nvtx-marker \
  --disable-cuda-graph

# Terminal 2: Profile the benchmarking client
nsys profile --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  -o layerwise_profile \
  python -m sglang.bench_serving --backend sglang --num-prompts 10
```

This approach profiles the entire client execution, including all server interactions. The layerwise NVTX markers will be visible in the Nsight Systems timeline.

**中文对照**：**方法 2：更简单的方法，不使用 `/start_profile` API**

对于不需要精细控制分析开始/停止的更简单用例，你可以使用 Nsight Systems 分析整个工作负载：

此方法分析整个客户端执行，包括所有服务器交互。逐层 NVTX 标记将在 Nsight Systems 时间线中可见。

**Viewing the profiling results:**

Open the generated `.qdrep` file with Nsight Systems:

```bash
nsys-ui layerwise_profile.qdrep
```

In the Nsight Systems GUI, you'll see:
- **NVTX ranges**: Each layer appears as a labeled range in the timeline with detailed information in the marker metadata
- **CUDA kernels**: All GPU kernels are shown alongside the layer annotations
- **Layer hierarchy**: The full module path (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct.model.layers.0.self_attn.qkv_proj`) helps identify specific layers. The prefix uses the full model path from `--model-path`.
- **Tensor shapes**: Input/output dimensions and parameter shapes are included in the NVTX marker data

**中文对照**：**查看分析结果：**

使用 Nsight Systems 打开生成的 `.qdrep` 文件：

在 Nsight Systems GUI 中，你将看到：
- **NVTX 范围**：每一层在时间线中显示为带标签的范围，标记元数据中包含详细信息
- **CUDA 内核**：所有 GPU 内核都显示在层注释旁边
- **层层次结构**：完整的模块路径（例如 `meta-llama/Meta-Llama-3.1-8B-Instruct.model.layers.0.self_attn.qkv_proj`）有助于识别特定层。前缀使用 `--model-path` 中的完整模型路径
- **张量形状**：输入/输出维度和参数形状包含在 NVTX 标记数据中

**Benefits of layerwise NVTX profiling:**

- **Granular visibility**: See exactly which layers are taking the most time
- **Memory tracking**: Identify layers with large memory allocations
- **Bottleneck identification**: Quickly locate inefficient operations
- **Communication overhead**: In multi-GPU setups, see per-layer communication costs
- **Development debugging**: Validate that model架构 changes have the expected performance impact

**中文对照**：**逐层 NVTX 分析的好处：**

- **粒度可见性**：准确查看哪些层花费最多时间
- **内存跟踪**：识别具有大内存分配的层
- **瓶颈识别**：快速定位低效操作
- **通信开销**：在多 GPU 设置中，查看每层通信成本
- **开发调试**：验证模型架构更改是否具有预期的性能影响

## Other tips

1. You can benchmark a model using dummy weights by only providing the config.json file. This allows for quick testing of model variants without training. To do so, add `--load-format dummy` to the above commands and then you only need a correct `config.json` under the checkpoint folder.
2. You can benchmark a model with modified configs (e.g., less layers) by using `--json-model-override-args`. For example, you can benchmark a model with only 2 layers and 2 kv heads using:

   ```bash
   python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch 32 --input-len 256 --output-len 32 --load-format dummy --json-model-override-args '{"num_hidden_layers": 1, "num_key_value_heads": 1}'
   ```

3. You can use `--python-backtrace=cuda` to see python call stack for all CUDA kernels, as in PyTorch Profiler. (Caveat: this can cause inaccurately long kernel runtimes for CUDA event based timing)
4. For more arguments see [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

**中文对照**：## 其他技巧

1. 你可以通过仅提供 config.json 文件来使用虚拟权重对模型进行基准测试。这允许在不训练的情况下快速测试模型变体。为此，将 `--load-format dummy` 添加到上述命令，然后你只需要在 checkpoint 文件夹下有一个正确的 `config.json`。
2. 你可以使用 `--json-model-override-args` 对修改了配置的模型进行基准测试（例如，较少的层）。例如，你可以对只有 2 层和 2 个 kv head 的模型进行基准测试：
3. 你可以使用 `--python-backtrace=cuda` 查看所有 CUDA 内核的 Python 调用堆栈，如 PyTorch Profiler 中所示。（注意：这可能会导致基于 CUDA 事件计时的内核运行时间不准确地变长）
4. 有关更多参数，请参阅 [Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/managers/scheduler_profiler_mixin.py` | PyTorch Profiler 集成：`init_profiler()`、步数计数、通过 HTTP API 启动/停止 |
| `python/sglang/srt/entrypoints/http_server.py` | `/start_profile`、`/end_profile` HTTP 端点用于编程式分析器控制 |
| `python/sglang/bench_one_batch.py` | 无服务器的单批次延迟基准测试（静态批次，无动态批处理） |
| `python/sglang/bench_offline_throughput.py` | 离线吞吐量基准测试，支持引擎级分析 |
| `python/sglang/bench_serving.py` | 在线服务基准测试：TTFT、ITL、吞吐量、并发指标 |

### 关键代码逻辑

- **分析器生命周期**：`scheduler_profiler_mixin.py` 通过 `init_profiler()` 管理 PyTorch Profiler，可配置 `start_step`、`num_steps` 和 `activities`（CPU/GPU/CUDA_PROFILER）
- **HTTP 分析器 API**：`/start_profile` 触发 `cudaProfilerStart()` 用于 Nsight 集成；`/end_profile` 停止并保存跟踪文件
- **跟踪合并**：分布式跟踪合并器生成每 rank 文件（`TP-{tp}-DP-{dp}-PP-{pp}-EP-{ep}.trace.json.gz`）和合并输出
- **NVTX 逐层**：`--enable-layerwise-nvtx-marker` 为每个模型层添加 NVTX 注释，用于 Nsight Systems 可视化

### 集成要点

- **环境变量**：`SGLANG_TORCH_PROFILER_DIR`（跟踪输出路径）、`SGLANG_PROFILE_WITH_STACK`（禁用堆栈跟踪以提高兼容性）
- **服务器标志**：`--enable-layerwise-nvtx-marker`、`--disable-cuda-graph`（用于准确的 NVTX 分析）
- **基准测试标志**：`--profile`（启用分析）、`--flush-cache`（基准测试前清除 RadixCache）
