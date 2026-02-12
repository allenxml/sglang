# AMD GPUs

This document describes how to run SGLang on AMD GPUs. If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

**中文对照**：本文档介绍如何在 AMD GPU 上运行 SGLang。如果您遇到问题或有疑问，请[提交 issue](https://github.com/sgl-project/sglang/issues)。

## System Configuration

When using AMD GPUs (such as MI300X), certain system-level optimizations help ensure stable performance. Here we take MI300X as an example. AMD provides official documentation for MI300X optimization and system tuning:

- [AMD MI300X Tuning Guides](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)
- [LLM inference performance validation on AMD Instinct MI300X](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/vllm-benchmark.html)
- [AMD Instinct MI300X System Optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html)
- [AMD Instinct MI300X Workload Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
- [Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)

**中文对照**：使用 AMD GPU（如 MI300X）时，某些系统级优化有助于确保稳定性能。这里以 MI300X 为例。AMD 提供了 MI300X 优化和系统调优的官方文档：

- [AMD MI300X 调优指南](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)
- [AMD Instinct MI300X 上的 LLM 推理性能验证](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/vllm-benchmark.html)
- [AMD Instinct MI300X 系统优化](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html)
- [AMD Instinct MI300X 工作负载优化](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
- [在 AMD Instinct MI300X 上加速 DeepSeek-R1 推理](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)

**NOTE:** We strongly recommend reading these docs and guides entirely to fully utilize your system.

**中文对照**：**注意：** 我们强烈建议您完整阅读这些文档和指南，以充分利用您的系统。

Below are a few key settings to confirm or enable for SGLang:

**中文对照**：以下是 SGLang 需要确认或启用的几个关键设置：

### Update GRUB Settings

In `/etc/default/grub`, append the following to `GRUB_CMDLINE_LINUX`:

```text
pci=realloc=off iommu=pt
```

Afterward, run `sudo update-grub` (or your distro's equivalent) and reboot.

**中文对照**：### 更新 GRUB 设置

在 `/etc/default/grub` 中，将以下内容追加到 `GRUB_CMDLINE_LINUX`：

```text
pci=realloc=off iommu=pt
```

然后运行 `sudo update-grub`（或您发行版的等效命令）并重启。

### Disable NUMA Auto-Balancing

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

You can automate or verify this change using [this helpful script](https://github.com/ROCm/triton/blob/rocm_env/scripts/amd/env_check.sh).

**中文对照**：### 禁用 NUMA 自动平衡

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

您可以使用[这个有用的脚本](https://github.com/ROCm/triton/blob/rocm_env/scripts/amd/env_check.sh)来自动化或验证此更改。

Again, please go through the entire documentation to confirm your system is using the recommended configuration.

**中文对照**：同样，请完整阅读文档以确认您的系统使用的是推荐配置。

## Install SGLang

You can install SGLang using one of the methods below.

**中文对照**：## 安装 SGLang

您可以使用以下方法之一安装 SGLang。

### Install from Source

```bash
# Use the last release branch
git clone -b v0.5.6.post2 https://github.com/sgl-project/sglang.git
cd sglang

# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_rocm.py install

# Install sglang python package along with diffusion support
cd ..
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_hip]"
```

**中文对照**：### 从源码安装

```bash
# 使用最新的发布分支
git clone -b v0.5.6.post2 https://github.com/sgl-project/sglang.git
cd sglang

# 编译 sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_rocm.py install

# 安装 sglang python 包及扩散模型支持
cd ..
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_hip]"
```

### Install Using Docker (Recommended)

The docker images are available on Docker Hub at [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags), built from [rocm.Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker).

The steps below show how to build and use an image.

**中文对照**：### 使用 Docker 安装（推荐）

Docker 镜像可在 Docker Hub 上的 [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags) 获取，构建自 [rocm.Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker)。

以下步骤展示如何构建和使用镜像。

1. Build the docker image.
   If you use pre-built images, you can skip this step and replace `sglang_image` with the pre-built image names in the steps below.

   ```bash
   docker build -t sglang_image -f rocm.Dockerfile .
   ```

2. Create a convenient alias.

   ```bash
   alias drun='docker run -it --rm --network=host --privileged --device=/dev/kfd --device=/dev/dri \
       --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       -v $HOME/dockerx:/dockerx \
       -v /data:/data'
   ```

   If you are using RDMA, please note that:
     - `--network host` and `--privileged` are required by RDMA. If you don't need RDMA, you can remove them.
     - You may need to set `NCCL_IB_GID_INDEX` if you are using RoCE, for example: `export NCCL_IB_GID_INDEX=3`.

**中文对照**：1. 构建 Docker 镜像。
   如果您使用预构建镜像，可以跳过此步骤，并在以下步骤中将 `sglang_image` 替换为预构建镜像名称。

   ```bash
   docker build -t sglang_image -f rocm.Dockerfile .
   ```

2. 创建便捷的别名。

   ```bash
   alias drun='docker run -it --rm --network=host --privileged --device=/dev/kfd --device=/dev/dri \
       --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       -v $HOME/dockerx:/dockerx \
       -v /data:/data'
   ```

   如果您使用 RDMA，请注意：
     - RDMA 需要 `--network host` 和 `--privileged`。如果不需要 RDMA，可以移除它们。
     - 如果您使用 RoCE，可能需要设置 `NCCL_IB_GID_INDEX`，例如：`export NCCL_IB_GID_INDEX=3`。

3. Launch the server.

   **NOTE:** Replace `<secret>` below with your [huggingface hub token](https://huggingface.co/docs/hub/en/security-tokens).

   ```bash
   drun -p 30000:30000 \
       -v ~/.cache/huggingface:/root/.cache/huggingface \
       --env "HF_TOKEN=<secret>" \
       sglang_image \
       python3 -m sglang.launch_server \
       --model-path NousResearch/Meta-Llama-3.1-8B \
       --host 0.0.0.0 \
       --port 30000
   ```

**中文对照**：3. 启动服务器。

   **注意：** 将下面的 `<secret>` 替换为您的 [huggingface hub token](https://huggingface.co/docs/hub/en/security-tokens)。

   ```bash
   drun -p 30000:30000 \
       -v ~/.cache/huggingface:/root/.cache/huggingface \
       --env "HF_TOKEN=<secret>" \
       sglang_image \
       python3 -m sglang.launch_server \
       --model-path NousResearch/Meta-Llama-3.1-8B \
       --host 0.0.0.0 \
       --port 30000
   ```

4. To verify the utility, you can run a benchmark in another terminal or refer to [other docs](https://docs.sglang.io/basic_usage/openai_api_completions.html) to send requests to the engine.

   ```bash
   drun sglang_image \
       python3 -m sglang.bench_serving \
       --backend sglang \
       --dataset-name random \
       --num-prompts 4000 \
       --random-input 128 \
       --random-output 128
   ```

With your AMD system properly configured and SGLang installed, you can now fully leverage AMD hardware to power SGLang's machine learning capabilities.

**中文对照**：4. 要验证功能，您可以在另一个终端运行基准测试，或参考[其他文档](https://docs.sglang.io/basic_usage/openai_api_completions.html)向引擎发送请求。

   ```bash
   drun sglang_image \
       python3 -m sglang.bench_serving \
       --backend sglang \
       --dataset-name random \
       --num-prompts 4000 \
       --random-input 128 \
       --random-output 128
   ```

   配置好 AMD 系统并安装 SGLang 后，您现在可以充分利用 AMD 硬件来支持 SGLang 的机器学习功能。

## Examples

**中文对照**：## 示例

### Running DeepSeek-V3

The only difference when running DeepSeek-V3 is in how you start the server. Here's an example command:

**中文对照**：### 运行 DeepSeek-V3

运行 DeepSeek-V3 的唯一区别在于启动服务器的方式。以下是示例命令：

```bash
drun -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    --env "HF_TOKEN=<secret>" \
    sglang_image \
    python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \ # <- here
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000
```

[Running DeepSeek-R1 on a single NDv5 MI300X VM](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/running-deepseek-r1-on-a-single-ndv5-mi300x-vm/4372726) could also be a good reference.

### Running Llama3.1

Running Llama3.1 is nearly identical to running DeepSeek-V3. The only difference is in the model specified when starting the server, shown by the following example command:

**中文对照**：[在单个 NDv5 MI300X VM 上运行 DeepSeek-R1](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/running-deepseek-r1-on-a-single-ndv5-mi300x-vm/4372726) 也可以作为很好的参考。

### 运行 Llama3.1

运行 Llama3.1 与运行 DeepSeek-V3 几乎完全相同。唯一的区别在于启动服务器时指定的模型，如以下示例命令所示：

```bash
drun -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    --env "HF_TOKEN=<secret>" \
    sglang_image \
    python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \ # <- here
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000
```

### Warmup Step

When the server displays `The server is fired up and ready to roll!`, it means the startup is successful.

**中文对照**：### 预热步骤

当服务器显示 `The server is fired up and ready to roll!` 时，表示启动成功。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py` | 针对 DeepSeek 模型的 ROCm 优化 MLA 解码内核（带 RoPE） |
| `python/sglang/srt/layers/moe/rocm_moe_utils.py` | ROCm 专用 MoE 工具（专家路由、AITER 集成） |
| `python/sglang/srt/layers/utils/multi_platform.py` | 平台抽象层：`is_hip()` 检测、设备特定分发 |
| `python/sglang/srt/utils/rpd_utils.py` | ROCm Profiler Data (RPD) 工具用于性能分析 |
| `python/sglang/srt/distributed/device_communicators/pymscclpp.py` | AMD GPU 集合操作的 MSCCLPP 通信后端 |
| `sgl-kernel/setup_rocm.py` | ROCm 专用内核编译设置（HIP 工具链） |
| `docker/rocm.Dockerfile` | AMD GPU Docker 镜像构建配置 |

### 关键代码逻辑

- **平台检测**：`multi_platform.py` 中的 `is_hip()` 在整个代码库中控制 ROCm 专用代码路径
- **注意力分发**：ROCm 使用基于 Triton 的注意力内核而非 FlashInfer；MLA 解码具有专用的 ROCm 内核
- **MoE 加速**：`rocm_moe_utils.py` 集成 AITER（AMD Inference Toolkit Engine for ROCm）以优化专家计算
- **sgl-kernel**：独立的 `setup_rocm.py` 编译 HIP 内核；通过 `python setup_rocm.py install` 安装

### 集成要点

- **安装**：ROCm 路径使用 `pyproject_other.toml` 和 `pip install -e "python[all_hip]"` 安装 HIP 专用依赖
- **Docker**：`rocm.Dockerfile` 基于 ROCm 基础镜像构建，预装所有 AMD 专用库
- **分析**：RPD 工具通过 `rpd_utils.py` 启用 ROCm 原生分析；兼容 AMD 的分析工具
