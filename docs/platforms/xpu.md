# XPU

The document addresses how to set up the [SGLang](https://github.com/sgl-project/sglang) environment and run LLM inference on Intel GPU, [see more context about Intel GPU support within PyTorch ecosystem](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).

Specifically, SGLang is optimized for [Intel® Arc™ Pro B-Series Graphics](https://www.intel.com/content/www/us/en/ark/products/series/242616/intel-arc-pro-b-series-graphics.html) and [
Intel® Arc™ B-Series Graphics](https://www.intel.com/content/www/us/en/ark/products/series/240391/intel-arc-b-series-graphics.html).

**中文对照**：# XPU

本文档介绍如何设置 [SGLang](https://github.com/sgl-project/sglang) 环境并在 Intel GPU 上运行 LLM 推理，[了解更多关于 PyTorch 生态系统中 Intel GPU 支持的信息](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)。

具体来说，SGLang 针对 [Intel® Arc™ Pro B-Series 显卡](https://www.intel.com/content/www/us/en/ark/products/series/242616/intel-arc-pro-b-series-graphics.html) 和 [Intel® Arc™ B-Series 显卡](https://www.intel.com/content/www/us/en/ark/products/series/240391/intel-arc-b-series-graphics.html) 进行了优化。

## Optimized Model List

A list of LLMs have been optimized on Intel GPU, and more are on the way:

| Model Name | BF16 |
|:---:|:---:|
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Qwen2.5-1.5B |   [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |

**Note:** The model identifiers listed in the table above
have been verified on [Intel® Arc™ B580 Graphics](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html).

**中文对照**：## 优化模型列表

已在 Intel GPU 上优化了一批 LLM，更多模型正在优化中：

| 模型名称 | BF16 |
|:---:|:---:|
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Qwen2.5-1.5B |   [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |

**注意**：上表中列出的模型标识符已在 [Intel® Arc™ B580 显卡](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html) 上验证。

## Installation

### Install From Source

Currently SGLang XPU only supports installation from source. Please refer to ["Getting Started on Intel GPU"](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) to install XPU dependency.

```bash
# Create and activate a conda environment
conda create -n sgl-xpu python=3.12 -y
conda activate sgl-xpu

# Set PyTorch XPU as primary pip install channel to avoid installing the larger CUDA-enabled version and prevent potential runtime issues.
pip3 install torch==2.9.0+xpu torchao torchvision torchaudio pytorch-triton-xpu==3.5.0 --index-url https://download.pytorch.org/whl/xpu
pip3 install xgrammar --no-deps # xgrammar will introduce CUDA-enabled triton which might conflict with XPU

# Clone the SGLang code
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout <YOUR-DESIRED-VERSION>

# Use dedicated toml file
cd python
cp pyproject_xpu.toml pyproject.toml
# Install SGLang dependent libs, and build SGLang main package
pip install --upgrade pip setuptools
pip install -v .
```

### Install Using Docker

The docker for XPU is under active development. Please stay tuned.

**中文对照**：## 安装

### 从源码安装

目前 SGLang XPU 仅支持从源码安装。请参考 ["Intel GPU 入门指南"](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) 安装 XPU 依赖。

```bash
# 创建并激活 conda 环境
conda create -n sgl-xpu python=3.12 -y
conda activate sgl-xpu

# 将 PyTorch XPU 设置为主要 pip 安装通道，以避免安装更大的 CUDA 启用版本并防止潜在的运行时问题。
pip3 install torch==2.9.0+xpu torchao torchvision torchaudio pytorch-triton-xpu==3.5.0 --index-url https://download.pytorch.org/whl/xpu
pip3 install xgrammar --no-deps # xgrammar 会引入支持 CUDA 的 triton，可能与 XPU 冲突

# 克隆 SGLang 代码
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout <YOUR-DESIRED-VERSION>

# 使用专用的 toml 文件
cd python
cp pyproject_xpu.toml pyproject.toml
# 安装 SGLang 依赖库，并构建 SGLang 主包
pip install --upgrade pip setuptools
pip install -v .
```

### 使用 Docker 安装

XPU 的 Docker 正在积极开发中。请持续关注。

## Launch of the Serving Engine

Example command to launch SGLang serving:

```bash
python -m sglang.launch_server       \
    --model <MODEL_ID_OR_PATH>       \
    --trust-remote-code              \
    --disable-overlap-schedule       \
    --device xpu                     \
    --host 0.0.0.0                   \
    --tp 2                           \   # using multi GPUs
    --attention-backend intel_xpu    \   # using intel optimized XPU attention backend
    --page-size                      \   # intel_xpu attention backend supports [32, 64, 128]
```

## Benchmarking with Requests

You can benchmark the performance via the `bench_serving` script.
Run the command in another terminal.

```bash
python -m sglang.bench_serving   \
    --dataset-name random        \
    --random-input-len 1024      \
    --random-output-len 1024     \
    --num-prompts 1              \
    --request-rate inf           \
    --random-range-ratio 1.0
```

The detail explanations of the parameters can be looked up by the command:

```bash
python -m sglang.bench_serving -h
```

Additionally, the requests can be formed with
[OpenAI Completions API](https://docs.sglang.io/basic_usage/openai_api_completions.html)
and sent via the command line (e.g. using `curl`) or via your own script.

**中文对照**：## 启动服务引擎

启动 SGLang 服务的示例命令：

```bash
python -m sglang.launch_server       \
    --model <MODEL_ID_OR_PATH>       \
    --trust-remote-code              \
    --disable-overlap-schedule       \
    --device xpu                     \
    --host 0.0.0.0                   \
    --tp 2                           \   # 使用多 GPU
    --attention-backend intel_xpu    \   # 使用英特尔优化的 XPU 注意力后端
    --page-size                      \   # intel_xpu 注意力后端支持 [32, 64, 128]
```

## 使用请求进行基准测试

您可以通过 `bench_serving` 脚本对性能进行基准测试。
在另一个终端中运行该命令。

```bash
python -m sglang.bench_serving   \
    --dataset-name random        \
    --random-input-len 1024      \
    --random-output-len 1024     \
    --num-prompts 1              \
    --request-rate inf           \
    --random-range-ratio 1.0
```

参数的详细说明可以通过以下命令查看：

```bash
python -m sglang.bench_serving -h
```

此外，请求可以使用 [OpenAI Completions API](https://docs.sglang.io/basic_usage/openai_api_completions.html)
进行格式化，并通过命令行（例如使用 `curl`）或您自己的脚本发送。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/layers/attention/xpu_backend.py` | Intel XPU 注意力后端：Intel Arc/数据中心 GPU 的优化注意力内核 |
| `python/sglang/srt/distributed/device_communicators/xpu_communicator.py` | XPU 集合通信：多 GPU Intel 设置的 allreduce/allgather |
| `python/sglang/srt/configs/device_config.py` | 设备检测：设置 `--device xpu` 时路由到 XPU 后端 |
| `python/sglang/srt/layers/utils/multi_platform.py` | 平台抽象：XPU 检测和设备特定分发 |

### 集成要点

- **安装**：使用 `pyproject_xpu.toml` 配合 PyTorch XPU wheel（`torch+xpu`、`pytorch-triton-xpu`）
- **注意力后端**：`--attention-backend intel_xpu` 启用优化的 XPU 注意力，页大小为 32、64 或 128
- **多 GPU**：`--tp 2` 通过 `xpu_communicator.py` 在多个 Intel GPU 上分布模型
- **依赖**：使用 `--no-deps` 安装 `xgrammar` 以避免引入支持 CUDA 的 triton
