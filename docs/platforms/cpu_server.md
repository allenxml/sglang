# CPU Servers

The document addresses how to set up the [SGLang](https://github.com/sgl-project/sglang) environment and run LLM inference on CPU servers.
SGLang is enabled and optimized on the CPUs equipped with Intel® AMX® Instructions,
which are 4th generation or newer Intel® Xeon® Scalable Processors.

**中文对照**：# CPU 服务器

本文档介绍如何设置 [SGLang](https://github.com/sgl-project/sglang) 环境并在 CPU 服务器上运行 LLM 推理。SGLang 在配备 Intel® AMX® 指令的 CPU 上启用并进行了优化，这些 CPU 是第 4 代或更新的 Intel® Xeon® 可扩展处理器。

## Optimized Model List

A list of popular LLMs are optimized and run efficiently on CPU,
including the most notable open-source models like Llama series, Qwen series,
and DeepSeek series like DeepSeek-R1 and DeepSeek-V3.1-Terminus.

| Model Name | BF16 | W8A8_INT8 | FP8 |
|:---:|:---:|:---:|:---:|
| DeepSeek-R1 |   | [meituan/DeepSeek-R1-Channel-INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8) | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| DeepSeek-V3.1-Terminus |   | [IntervitensInc/DeepSeek-V3.1-Terminus-Channel-int8](https://huggingface.co/IntervitensInc/DeepSeek-V3.1-Terminus-Channel-int8) | [deepseek-ai/DeepSeek-V3.1-Terminus](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus) |
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [RedHatAI/Llama-3.2-3B-quantized.w8a8](https://huggingface.co/RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8) |   |
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8) |   |
| QwQ-32B |   | [RedHatAI/QwQ-32B-quantized.w8a8](https://huggingface.co/RedHatAI/QwQ-32B-quantized.w8a8) |   |
| DeepSeek-Distilled-Llama |   | [RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8](https://huggingface.co/RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8) |   |
| Qwen3-235B |   |   | [Qwen/Qwen3-235B-A22B-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8) |

**Note:** The model identifiers listed in the table above
have been verified on 6th Gen Intel® Xeon® P-core platforms.

**中文对照**：**注意**：上表中列出的模型标识符已在第 6 代 Intel® Xeon® P-core 平台上验证。

## Installation

### Install Using Docker

It is recommended to use Docker for setting up the SGLang environment.
A [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/xeon.Dockerfile) is provided to facilitate the installation.
Replace `<secret>` below with your [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
# Clone the SGLang repository
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build the docker image
docker build -t sglang-cpu:latest -f xeon.Dockerfile .

# Initiate a docker container
docker run \
    -it \
    --privileged \
    --ipc=host \
    --network=host \
    -v /dev/shm:/dev/shm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 30000:30000 \
    -e "HF_TOKEN=<secret>" \
    sglang-cpu:latest /bin/bash
```

### Install From Source

If you prefer to install SGLang in a bare metal environment,
the setup process is as follows:

Please install the required packages and libraries beforehand if
they are not already present on your system.
You can refer to the Ubuntu-based installation commands in
[the Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/xeon.Dockerfile#L11)
for guidance.

1. Install `uv` package manager, then create and activate a virtual environment:

```bash
# Taking '/opt' as the example uv env folder, feel free to change it as needed
cd /opt
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12
source .venv/bin/activate
```

2. Create a config file to direct the installation channel
    (a.k.a. index-url) of `torch` related packages:

```bash
vim .venv/uv.toml
```

Press 'a' to enter insert mode of `vim`, paste the following content into the created file

```file
[[index]]
name = "torch"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "torchvision"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "torchaudio"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "triton"
url = "https://download.pytorch.org/whl/cpu"

```

Save the file (in `vim`, press 'esc' to exit insert mode, then ':x+Enter'),
and set it as the default `uv` config.

```bash
export UV_CONFIG_FILE=/opt/.venv/uv.toml
```

3. Clone the `sglang` source code and build the packages

```bash
# Clone the SGLang code
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout <YOUR-DESIRED-VERSION>

# Use dedicated toml file
cd python
cp pyproject_cpu.toml pyproject.toml
# Install SGLang dependent libs, and build SGLang main package
uv pip install --upgrade pip setuptools
uv pip install .
<<<<<<< HEAD
=======
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 torchao==0.14.1 triton==3.5.0 --force-reinstall
>>>>>>> v0.5.8

# Build the CPU backend kernels
cd ../sgl-kernel
cp pyproject_cpu.toml pyproject.toml
uv pip install .
```

4. Set the required environment variables

```bash
export SGLANG_USE_CPU_ENGINE=1

# Set 'LD_LIBRARY_PATH' and 'LD_PRELOAD' to ensure the libs can be loaded by sglang processes
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=${LD_PRELOAD}:/opt/.venv/lib/libiomp5.so:${LD_LIBRARY_PATH}/libtcmalloc.so.4:${LD_LIBRARY_PATH}/libtbbmalloc.so.2
```

Notes:

- Note that the environment variable `SGLANG_USE_CPU_ENGINE=1`
    is required to enable the SGLang service with the CPU engine.

- If you encounter code compilation issues during the `sgl-kernel` building process,
    please check your `gcc` and `g++` versions and upgrade them if they are outdated.
    It is recommended to use `gcc-13` and `g++-13` as they have been verified
    in the official Docker container.

- The system library path is typically located in one of the following directories:
    `~/.local/lib/`, `/usr/local/lib/`, `/usr/local/lib64/`, `/usr/lib/`, `/usr/lib64/`
    and `/usr/lib/x86_64-linux-gnu/`. In the above example commands, `/usr/lib/x86_64-linux-gnu`
    is used. Please adjust the path according to your server configuration.

- It is recommended to add the following to your `~/.bashrc` file to
    avoid setting these variables every time you open a new terminal:

    ```bash
    source .venv/bin/activate
    export SGLANG_USE_CPU_ENGINE=1
    export LD_LIBRARY_PATH=<YOUR-SYSTEM-LIBRARY-FOLDER>
    export LD_PRELOAD=<YOUR-LIBS-PATHS>
    ```

## Launch of the Serving Engine

Example command to launch SGLang serving:

```bash
python -m sglang.launch_server   \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --host 0.0.0.0               \
    --tp 6
```

Notes:

1. For running W8A8 quantized models, please add the flag `--quantization w8a8_int8`.

2. The flag `--tp 6` specifies that tensor parallelism will be applied using 6 ranks (TP6).
    The number of TP specified is how many TP ranks will be used during the execution.
    On a CPU platform, a TP rank means a sub-NUMA cluster (SNC).
    Usually we can get the SNC information (How many available) from the Operating System with e.g. `lscpu` command.

    If the specified TP rank number differs from the total SNC count,
    the system will automatically utilize the first `n` SNCs.
    Note that `n` cannot exceed the total SNC number, doing so will result in an error.

    To specify the cores to be used, we need to explicitly set the environment variable `SGLANG_CPU_OMP_THREADS_BIND`.
    For example, if we want to run the SGLang service using the first 40 cores of each SNC on a Xeon® 6980P server,
    which has 43-43-42 cores on the 3 SNCs of a socket, we should set:

    ```bash
    export SGLANG_CPU_OMP_THREADS_BIND="0-39|43-82|86-125|128-167|171-210|214-253"
    ```

    Please beware that with SGLANG_CPU_OMP_THREADS_BIND set,
    the available memory amounts of the ranks may not be determined in prior.
    You may need to set proper `--max-total-tokens` to avoid the out-of-memory error.

3. For optimizing decoding with torch.compile, please add the flag `--enable-torch-compile`.
    To specify the maximum batch size when using `torch.compile`, set the flag `--torch-compile-max-bs`.
    For example, `--enable-torch-compile --torch-compile-max-bs 4` means using `torch.compile`
    and setting the maximum batch size to 4. Currently the maximum applicable batch size
    for optimizing with `torch.compile` is 16.

4. A warmup step is automatically triggered when the service is started.
    The server is ready when you see the log `The server is fired up and ready to roll!`.

## Benchmarking with Requests

You can benchmark the performance via the `bench_serving` script.
Run the command in another terminal. An example command would be:

```bash
python -m sglang.bench_serving   \
    --dataset-name random        \
    --random-input-len 1024      \
    --random-output-len 1024     \
    --num-prompts 1              \
    --request-rate inf           \
    --random-range-ratio 1.0
```

Detailed parameter descriptions are available via the command:

```bash
python -m sglang.bench_serving -h
```

Additionally, requests can be formatted using
[the OpenAI Completions API](https://docs.sglang.io/basic_usage/openai_api_completions.html)
and sent via the command line (e.g., using `curl`) or through your own scripts.

## Example Usage Commands

Large Language Models can range from fewer than 1 billion to several hundred billion parameters.
Dense models larger than 20B are expected to run on flagship 6th Gen Intel® Xeon® processors
with dual sockets and a total of 6 sub-NUMA clusters. Dense models of approximately 10B parameters or fewer,
or MoE (Mixture of Experts) models with fewer than 10B activated parameters, can run on more common
4th generation or newer Intel® Xeon® processors, or utilize a single socket of the flagship 6th Gen Intel® Xeon® processors.

### Example: Running DeepSeek-V3.1-Terminus

An example command to launch service of W8A8_INT8 DeepSeek-V3.1-Terminus on a Xeon® 6980P server:

```bash
python -m sglang.launch_server                                 \
    --model IntervitensInc/DeepSeek-V3.1-Terminus-Channel-int8 \
    --trust-remote-code                                        \
    --disable-overlap-schedule                                 \
    --device cpu                                               \
    --quantization w8a8_int8                                   \
    --host 0.0.0.0                                             \
    --enable-torch-compile                                     \
    --torch-compile-max-bs 4                                   \
    --tp 6
```

Similarly, an example command to launch service of FP8 DeepSeek-V3.1-Terminus would be:

```bash
python -m sglang.launch_server                     \
    --model deepseek-ai/DeepSeek-V3.1-Terminus     \
    --trust-remote-code                            \
    --disable-overlap-schedule                     \
    --device cpu                                   \
    --host 0.0.0.0                                 \
    --enable-torch-compile                         \
    --torch-compile-max-bs 4                       \
    --tp 6
```

Note: Please set `--torch-compile-max-bs` to the maximum desired batch size for your deployment,
which can be up to 16. The value `4` in the examples is illustrative.

### Example: Running Llama-3.2-3B

An example command to launch service of Llama-3.2-3B with BF16 precision:

```bash
python -m sglang.launch_server                     \
    --model meta-llama/Llama-3.2-3B-Instruct       \
    --trust-remote-code                            \
    --disable-overlap-schedule                     \
    --device cpu                                   \
    --host 0.0.0.0                                 \
    --enable-torch-compile                         \
    --torch-compile-max-bs 16                      \
    --tp 2
```

The example command to launch service of W8A8_INT8 version of Llama-3.2-3B:

```bash
python -m sglang.launch_server                     \
    --model RedHatAI/Llama-3.2-3B-quantized.w8a8   \
    --trust-remote-code                            \
    --disable-overlap-schedule                     \
    --device cpu                                   \
    --quantization w8a8_int8                       \
    --host 0.0.0.0                                 \
    --enable-torch-compile                         \
    --torch-compile-max-bs 16                      \
    --tp 2
```

Note: The `--torch-compile-max-bs` and `--tp` settings are examples that should be adjusted for your setup.
For instance, use `--tp 3` to utilize 1 socket with 3 sub-NUMA clusters on an Intel® Xeon® 6980P server.

Once the server have been launched, you can test it using the `bench_serving` command or create
your own commands or scripts following [the benchmarking example](#benchmarking-with-requests).

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/model_executor/cpu_graph_runner.py` | CPU 图运行器：基于 torch.compile 的执行，替代 CUDA 图 |
| `python/sglang/srt/layers/attention/intel_amx_backend.py` | Intel AMX 注意力后端：针对第 4 代及以上 Xeon 处理器优化 |
| `python/sglang/srt/layers/amx_utils.py` | Intel 高级矩阵扩展的 AMX 工具函数 |
| `python/sglang/srt/mem_cache/memory_pool_host.py` | 主机（CPU）内存池：在系统 RAM 而非 GPU VRAM 中管理 KV 缓存 |
| `python/sglang/srt/configs/device_config.py` | 设备检测逻辑：设置 `--device cpu` 时路由到 CPU 后端 |
| `sgl-kernel/setup_cpu.py` | CPU 专用内核编译（不需要 CUDA） |
| `docker/xeon.Dockerfile` | 带 AMX/MKL 依赖的 Intel Xeon CPU Docker 镜像 |

### 关键代码逻辑

- **CPU 引擎激活**：环境变量 `SGLANG_USE_CPU_ENGINE=1` 在 `device_config.py` 中触发 CPU 专用代码路径
- **内存管理**：`memory_pool_host.py` 在系统 RAM 中分配 KV 缓存；TP rank 映射到子 NUMA 集群（SNCs）
- **torch.compile 优化**：`cpu_graph_runner.py` 使用 `torch.compile` 而非 CUDA 图进行解码加速
- **线程绑定**：`SGLANG_CPU_OMP_THREADS_BIND` 控制每 rank 的 OpenMP 线程亲和性以实现 NUMA 感知执行

### 集成要点

- **安装**：使用 `pyproject_cpu.toml`，从 `download.pytorch.org/whl/cpu` 获取仅 CPU 的 PyTorch wheel
- **CPU 上的张量并行**：TP rank 对应子 NUMA 集群；`--tp 6` 使用 6 个 SNCs
- **量化**：通过 `--quantization w8a8_int8` 的 W8A8 INT8 利用 Intel AMX VNNI 指令
- **torch.compile**：`--enable-torch-compile --torch-compile-max-bs N` 启用编译的解码内核（最大批大小最多 16）
