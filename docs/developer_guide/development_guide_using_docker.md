# Development Guide Using Docker

## Setup VSCode on a Remote Host
(Optional - you can skip this step if you plan to run sglang dev container locally)

**中文对照**：## 在远程主机上设置 VSCode
（可选 - 如果你计划在本地运行 sglang 开发容器，可以跳过此步骤）

1. In the remote host, download `code` from [Https://code.visualstudio.com/docs/?dv=linux64cli](https://code.visualstudio.com/download) and run `code tunnel` in a shell.

Example
```bash
wget https://vscode.download.prss.microsoft.com/dbazure/download/stable/fabdb6a30b49f79a7aba0f2ad9df9b399473380f/vscode_cli_alpine_x64_cli.tar.gz
tar xf vscode_cli_alpine_x64_cli.tar.gz

# https://code.visualstudio.com/docs/remote/tunnels
./code tunnel
```

**中文对照**：1. 在远程主机上，从 [Https://code.visualstudio.com/docs/?dv=linux64cli](https://code.visualstudio.com/download) 下载 `code`，并在 shell 中运行 `code tunnel`。

示例

2. In your local machine, press F1 in VSCode and choose "Remote Tunnels: Connect to Tunnel".

**中文对照**：2. 在本地机器上，按 F1 键在 VSCode 中，然后选择"远程隧道：连接到隧道"。

## Setup Docker Container

**中文对照**：## 设置 Docker 容器

### Option 1. Use the default dev container automatically from VSCode
There is a `.devcontainer` folder in the sglang repository root folder to allow VSCode to automatically start up within dev container. You can read more about this VSCode extension in VSCode official document [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers).
![image](https://github.com/user-attachments/assets/6a245da8-2d4d-4ea8-8db1-5a05b3a66f6d)
(*Figure 1: Diagram from VSCode official documentation [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers).*)

To enable this, you only need to:

**中文对照**：### 选项 1. 自动使用 VSCode 的默认开发容器
sglang 仓库根文件夹中有一个 `.devcontainer` 文件夹，允许 VSCode 自动在开发容器中启动。你可以在 VSCode 官方文档 [在容器内开发](https://code.visualstudio.com/docs/devcontainers/containers) 中阅读更多关于此 VSCode 扩展的信息。

要启用此功能，你只需要：
1. Start Visual Studio Code and install [VSCode dev container extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Press F1, type and choose "Dev Container: Open Folder in Container.
3. Input the `sglang` local repo path in your machine and press enter.

The first time you open it in dev container might take longer due to docker pull and build. Once it's successful, you should set on your status bar at the bottom left displaying that you are in a dev container:

![image](https://github.com/user-attachments/assets/650bba0b-c023-455f-91f9-ab357340106b)

**中文对照**：第一次在开发容器中打开它可能需要更长时间，因为 docker pull 和 build。一旦成功，你应该在状态栏左下角看到显示你处于开发容器中：

Now when you run `sglang.launch_server` in the VSCode terminal or start debugging using F5, sglang server will be started in the dev container with all your local changes applied automatically:

![image](https://github.com/user-attachments/assets/748c85ba-7f8c-465e-8599-2bf7a8dde895)

**中文对照**：现在，当你在 VSCode 终端中运行 `sglang.launch_server` 或使用 F5 开始调试时，sglang 服务器将在开发容器中启动，并自动应用你的所有本地更改：


### Option 2. Start up containers manually (advanced)

**中文对照**：### 选项 2. 手动启动容器（高级）

The following startup command is an example for internal development by the SGLang team. You can **modify or add directory mappings as needed**, especially for model weight downloads, to prevent repeated downloads by different Docker containers.

❗️ **Note on RDMA**

    1. `--network host` and `--privileged` are required by RDMA. If you don't need RDMA, you can remove them but keeping them there does not harm. Thus, we enable these two flags by default in the commands below.
    2. You may need to set `NCCL_IB_GID_INDEX` if you are using RoCE, for example: `export NCCL_IB_GID_INDEX=3`.

**中文对照**：以下启动命令是 SGLang 团队内部开发的示例。你可以根据需要**修改或添加目录映射**，特别是对于模型权重下载，以防止不同 Docker 容器重复下载。

**关于 RDMA 的注意事项**

    1. `--network host` 和 `--privileged` 是 RDMA 必需的。如果你不需要 RDMA，可以删除它们，但保留它们也没有害处。因此，我们在以下命令中默认启用这两个标志。
    2. 如果你使用 RoCE，可能需要设置 `NCCL_IB_GID_INDEX`，例如：`export NCCL_IB_GID_INDEX=3`。

```bash
# Change the name to yours
docker run -itd --shm-size 32g --gpus all -v <volumes-to-mount> --ipc=host --network=host --privileged --name sglang_dev lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_dev /bin/zsh
```
Some useful volumes to mount are:
1. **Huggingface model cache**: mounting model cache can avoid re-download every time docker restarts. Default location on Linux is `~/.cache/huggingface/`.
2. **SGLang repository**: code changes in the SGLang local repository will be automatically synced to the .devcontainer.

**中文对照**：一些有用的挂载卷：
1. **Huggingface 模型缓存**：挂载模型缓存可以避免每次 docker 重启时重新下载。Linux 上的默认位置是 `~/.cache/huggingface/`。
2. **SGLang 仓库**：SGLang 本地仓库中的代码更改将自动同步到 .devcontainer。

Example 1: Mounting local cache folder `/opt/dlami/nvme/.cache` but not the SGLang repo. Use this when you prefer to manually transfer local code changes to the devcontainer.
```bash
docker run -itd --shm-size 32g --gpus all -v /opt/dlami/nvme/.cache:/root/.cache --ipc=host --network=host --privileged --name sglang_zhyncs lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_zhyncs /bin/zsh
```
Example 2: Mounting both HuggingFace cache and local SGLang repo. Local code changes are automatically synced to the devcontainer as the SGLang is installed in editable mode in the dev image.
```bash
docker run -itd --shm-size 32g --gpus all -v $HOME/.cache/huggingface/:/root/.cache/huggingface -v $HOME/src/sglang:/sgl-workspace/sglang --ipc=host --network=host --privileged --name sglang_zhyncs lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_zhyncs /bin/zsh
```

**中文对照**：示例 1：挂载本地缓存文件夹 `/opt/dlami/nvme/.cache` 但不挂载 SGLang 仓库。当你更喜欢手动将本地代码更改传输到 devcontainer 时使用此选项。

示例 2：同时挂载 HuggingFace 缓存和本地 SGLang 仓库。本地代码更改会自动同步到 devcontainer，因为 SGLang 在开发镜像中以可编辑模式安装。

## Debug SGLang with VSCode Debugger

**中文对照**：## 使用 VSCode 调试器调试 SGLang
1. (Create if not exist) open `launch.json` in VSCode.
2. Add the following config and save. Please note that you can edit the script as needed to apply different parameters or debug a different program (e.g. benchmark script).
     ```JSON
       {
          "version": "0.2.0",
          "configurations": [
              {
                  "name": "Python Debugger: launch_server",
                  "type": "debugpy",
                  "request": "launch",
                  "module": "sglang.launch_server",
                  "console": "integratedTerminal",
                  "args": [
                      "--model-path", "meta-llama/Llama-3.2-1B",
                      "--host", "0.0.0.0",
                      "--port", "30000",
                      "--trust-remote-code",
                  ],
                  "justMyCode": false
              }
          ]
      }
    ```

3. Press "F5" to start. VSCode debugger will ensure that the program will pause at the breakpoints even if the program is running at remote SSH/Tunnel host + dev container.

**中文对照**：3. 按"F5"启动。VSCode 调试器将确保程序即使在远程 SSH/隧道主机 + dev 容器中运行，也会在断点处暂停。

## Profile

```bash
# Change batch size, input, output and add `disable-cuda-graph` (for easier analysis)
# e.g. DeepSeek V3
nsys profile -o deepseek_v3 python3 -m sglang.bench_one_batch --batch-size 1 --input 128 --output 256 --model deepseek-ai/DeepSeek-V3 --trust-remote-code --tp 8 --disable-cuda-graph
```

**中文对照**：## 性能分析

## Evaluation

```bash
# e.g. gsm8k 8 shot
python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8
```

**中文对照**：## 评估

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `.devcontainer/` | VSCode Dev Container 配置：Dockerfile + devcontainer.json 用于自动设置 |
| `docker/` | 生产和开发 Docker 镜像（CUDA、ROCm 变体） |
| `python/sglang/bench_one_batch.py` | 与 `nsys profile` 配合使用的单批次分析脚本 |
| `benchmark/gsm8k/bench_sglang.py` | GSM8K 准确性评估脚本 |

### 集成要点

- **Dev container**：VSCode 自动检测 `.devcontainer/` 并构建包含所有依赖的开发环境
- **Docker 卷**：挂载 `~/.cache/huggingface/` 用于模型缓存持久化；挂载 SGLang 仓库用于实时代码同步（可编辑安装）
- **VSCode 调试**：`launch.json` 配置，使用 `debugpy` 模块启动 `sglang.launch_server`
- **分析**：`nsys profile --cuda-graph-trace=node` 配合 `--disable-cuda-graph` 用于准确的内核级分析
