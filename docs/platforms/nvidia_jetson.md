# NVIDIA Jetson Orin

## Prerequisites

Before starting, ensure the following:

- [**NVIDIA Jetson AGX Orin Devkit**](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) is set up with **JetPack 6.1** or later.
- **CUDA Toolkit** and **cuDNN** are installed.
- Verify that the Jetson AGX Orin is in **high-performance mode**:
```bash
sudo nvpmodel -m 0
```
* * * * *

**中文对照**：# NVIDIA Jetson Orin

## 前置条件

在开始之前，请确保以下条件：

- 设置好 [**NVIDIA Jetson AGX Orin Devkit**](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)，并安装 **JetPack 6.1** 或更高版本。
- 已安装 **CUDA Toolkit** 和 **cuDNN**。
- 验证 Jetson AGX Orin 处于**高性能模式**：
```bash
sudo nvpmodel -m 0
```
* * * * *

## Installing and running SGLang with Jetson Containers
Clone the jetson-containers github repository:
```
git clone https://github.com/dusty-nv/jetson-containers.git
```
Run the installation script:
```
bash jetson-containers/install.sh
```
Build the container image:
```
jetson-containers build sglang
```
Run the container:
```
jetson-containers run $(autotag sglang)
```
Or you can also manually run a container with this command:
```
docker run --runtime nvidia -it --rm --network=host IMAGE_NAME
```
* * * * *

**中文对照**：## 使用 Jetson 容器安装和运行 SGLang
克隆 jetson-containers github 仓库：
```
git clone https://github.com/dusty-nv/jetson-containers.git
```
运行安装脚本：
```
bash jetson-containers/install.sh
```
构建容器镜像：
```
jetson-containers build sglang
```
运行容器：
```
jetson-containers run $(autotag sglang)
```
或者您也可以使用以下命令手动运行容器：
```
docker run --runtime nvidia -it --rm --network=host IMAGE_NAME
```
* * * * *

Running Inference
-----------------------------------------

Launch the server:
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --device cuda \
  --dtype half \
  --attention-backend flashinfer \
  --mem-fraction-static 0.8 \
  --context-length 8192
```
The quantization and limited context length (`--dtype half --context-length 8192`) are due to the limited computational resources in [Nvidia jetson kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/). A detailed explanation can be found in [Server Arguments](../advanced_features/server_arguments.md).

After launching the engine, refer to [Chat completions](https://docs.sglang.io/basic_usage/openai_api_completions.html#Usage) to test the usability.
* * * * *

**中文对照**：运行推理
-----------------------------------------

启动服务器：
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --device cuda \
  --dtype half \
  --attention-backend flashinfer \
  --mem-fraction-static 0.8 \
  --context-length 8192
```
量化 `--dtype half --context-length 8192` 是由于 [Nvidia jetson kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) 中有限的计算资源。详细说明请参阅[服务器参数](../advanced_features/server_arguments.md)。

启动引擎后，请参考[聊天补全](https://docs.sglang.io/basic_usage/openai_api_completions.html#Usage)来测试可用性。
* * * * *

Running quantization with TorchAO
-------------------------------------
TorchAO is suggested to NVIDIA Jetson Orin.
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --context-length 8192 \
    --torchao-config int4wo-128
```
This enables TorchAO's int4 weight-only quantization with a 128-group size. The usage of `--torchao-config int4wo-128` is also for memory efficiency.


* * * * *

Structured output with XGrammar
-------------------------------
Please refer to [SGLang doc structured output](../advanced_features/structured_outputs.ipynb).

* * * * *

Thanks to the support from [Nurgaliyev Shakhizat](https://github.com/shahizat), [Dustin Franklin](https://github.com/dusty-nv) and [Johnny Núñez Cano](https://github.com/johnnynunez).

References
----------
-   [NVIDIA Jetson AGX Orin Documentation](https://developer.nvidia.com/embedded/jetson-agx-orin)

**中文对照**：使用 TorchAO 运行量化
-------------------------------------
建议在 NVIDIA Jetson Orin 上使用 TorchAO。
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --context-length 8192 \
    --torchao-config int4wo-128
```
这将启用 TorchAO 的 int4 权重量化，分组大小为 128。使用 `--torchao-config int4wo-128` 也是为了提高内存效率。


* * * * *

使用 XGrammar 的结构化输出
-------------------------------
请参阅 [SGLang 结构化输出文档](../advanced_features/structured_outputs.ipynb)。

* * * * *

感谢 [Nurgaliyev Shakhizat](https://github.com/shahizat)、[Dustin Franklin](https://github.com/dusty-nv) 和 [Johnny Núñez Cano](https://github.com/johnnynunez) 的支持。

参考
----------
-   [NVIDIA Jetson AGX Orin 文档](https://developer.nvidia.com/embedded/jetson-agx-orin)

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/model_executor/model_runner.py` | ModelRunner：Jetson 上使用相同的 CUDA 前向传递（SM8.7 Orin 架构） |
| `python/sglang/srt/server_args.py` | ServerArgs：`--device cuda`、`--dtype half`、`--context-length`、`--torchao-config` 标志 |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | FlashInfer 注意力后端：Jetson 上的 `--attention-backend flashinfer` |
| `python/sglang/srt/constrained/xgrammar_backend.py` | XGrammar 结构化输出：在 Jetson 上支持 JSON/正则表达式约束 |

### 集成要点

- **无 Jetson 专用代码**：Jetson 运行标准 CUDA 代码路径；所有优化来自服务器参数
- **内存约束**：由于有限的 VRAM（32-64 GB 统一内存），需要 `--dtype half --context-length 8192 --mem-fraction-static 0.8`
- **量化**：推荐在边缘设备上使用 TorchAO `--torchao-config int4wo-128` 以提高内存效率
- **容器**：使用 `jetson-containers` 项目（`jetson-containers build sglang`）配合 NVIDIA 运行时
