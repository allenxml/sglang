# MindSpore Models

## Introduction

MindSpore is a high-performance AI framework optimized for Ascend NPUs. This doc guides users to run MindSpore models in SGLang.

**中文对照**：# MindSpore 模型

## 简介

MindSpore 是针对 Ascend NPU 优化的高性能 AI 框架。本文档指导用户在 SGLang 中运行 MindSpore 模型。

## Requirements

MindSpore currently only supports Ascend NPU devices. Users need to first install Ascend CANN software packages.
The CANN software packages can be downloaded from the [Ascend Official Website](https://www.hiascend.com). The recommended version is 8.3.RC2.

## Supported Models

Currently, the following models are supported:

- **Qwen3**: Dense and MoE models
- **DeepSeek V3/R1**
- *More models coming soon...*

**中文对照**：## 系统要求

MindSpore 目前仅支持 Ascend NPU 设备。用户需要首先安装 Ascend CANN 软件包。
CANN 软件包可以从 [Ascend 官方网站](https://www.hiascend.com) 下载。建议版本是 8.3.RC2。

## 支持的模型

目前支持以下模型：

- **Qwen3**：密集模型和 MoE 模型
- **DeepSeek V3/R1**
- *更多模型即将推出...*

## Installation

> **Note**: Currently, MindSpore models are provided by an independent package `sgl-mindspore`. Support for MindSpore is built upon current SGLang support for Ascend NPU platform. Please first [install SGLang for Ascend NPU](ascend_npu.md) and then install `sgl-mindspore`:

```shell
git clone https://github.com/mindspore-lab/sgl-mindspore.git
cd sgl-mindspore
pip install -e .
```

**中文对照**：## 安装

> **注意**：目前，MindSpore 模型由独立包 `sgl-mindspore` 提供。MindSpore 的支持建立在当前 SGLang 对 Ascend NPU 平台的支持之上。请首先[为 Ascend NPU 安装 SGLang](ascend_npu.md)，然后安装 `sgl-mindspore`：

```shell
git clone https://github.com/mindspore-lab/sgl-mindspore.git
cd sgl-mindspore
pip install -e .
```


## Run Model

Current SGLang-MindSpore supports Qwen3 and DeepSeek V3/R1 models. This doc uses Qwen3-8B as an example.

### Offline infer

Use the following script for offline infer:

```python
import sglang as sgl

# Initialize the engine with MindSpore backend
llm = sgl.Engine(
    model_path="/path/to/your/model",  # Local model path
    device="npu",                      # Use NPU device
    model_impl="mindspore",            # MindSpore implementation
    attention_backend="ascend",        # Attention backend
    tp_size=1,                         # Tensor parallelism size
    dp_size=1                          # Data parallelism size
)

# Generate text
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is"
]

sampling_params = {"temperature": 0, "top_p": 0.9}
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Generated: {output['text']}")
    print("---")
```

### Start server

Launch a server with MindSpore backend:

```bash
# Basic server startup
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --tp-size 1 \
    --dp-size 1
```

For distributed server with multiple nodes:

```bash
# Multi-node distributed server
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --dist-init-addr 127.0.0.1:29500 \
    --nnodes 2 \
    --node-rank 0 \
    --tp-size 4 \
    --dp-size 2
```

## Troubleshooting

#### Debug Mode

Enable sglang debug logging by log-level argument.

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --log-level DEBUG
```

Enable mindspore info and debug logging by setting environments.

```bash
export GLOG_v=1  # INFO
export GLOG_v=0  # DEBUG
```

#### Explicitly select devices

Use the following environment variable to explicitly select the devices to use.

```shell
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7  # to set device
```

#### Some communication environment issues

In case of some environment with special communication environment, users need set some environment variables.

```shell
export MS_ENABLE_LCCL=off # current not support LCCL communication mode in SGLang-MindSpore
```

#### Some dependencies of protobuf

In case of some environment with special protobuf version, users need set some environment variables to avoid binary version mismatch.

```shell
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # to avoid protobuf binary version mismatch
```

## Support
For MindSpore-specific issues:

- Refer to the [MindSpore documentation](https://www.mindspore.cn/)

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/mindspore.py` | MindSpore 模型包装器：将 MindSpore 模型适配到 SGLang 的前向接口 |
| `python/sglang/srt/model_executor/mindspore_runner.py` | MindSpore 运行器：在 Ascend NPU 上使用 MindSpore 后端执行前向传递 |
| `python/sglang/srt/server_args.py` | ServerArgs：`--model-impl mindspore` 和 `--attention-backend ascend` 标志 |

### 集成要点

- **独立包**：MindSpore 模型由 `sgl-mindspore` 包提供（从 `sgl-mindspore` 仓库安装）
- **服务器启动**：`--model-impl mindspore --device npu --attention-backend ascend` 激活 MindSpore 执行路径
- **分布式**：通过 `--dist-init-addr`、`--nnodes`、`--node-rank`、`--tp-size`、`--dp-size` 支持多节点
- **调试**：SGLang 使用 `--log-level DEBUG`；MindSpore 内部日志使用 `GLOG_v=1`（INFO）或 `GLOG_v=0`（DEBUG）
