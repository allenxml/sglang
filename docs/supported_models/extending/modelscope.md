# Use Models From ModelScope

To use a model from [ModelScope](https://www.modelscope.cn), set the environment variable `SGLANG_USE_MODELSCOPE`.

**中文对照**：要使用 ModelScope（https://www.modelscope.cn）中的模型，请设置环境变量 `SGLANG_USE_MODELSCOPE`。

```bash
export SGLANG_USE_MODELSCOPE=true
```

We take [Qwen2-7B-Instruct](https://www.modelscope.cn/models/qwen/qwen2-7b-instruct) as an example.

**中文对照**：我们以 [Qwen2-7B-Instruct](https://www.modelscope.cn/models/qwen/qwen2-7b-instruct) 为例。

Launch the Server:
```bash
python -m sglang.launch_server --model-path qwen/Qwen2-7B-Instruct --port 30000
```

Or start it by docker:

**中文对照**：或者通过 Docker 启动：

```bash
docker run --gpus all \
    -p 30000:30000 \
    -v ~/.cache/modelscope:/root/.cache/modelscope \
    --env "SGLANG_USE_MODELSCOPE=true" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000
```

Note that modelscope uses a different cache directory than huggingface. You may need to set it manually to avoid running out of disk space.

**中文对照**：请注意，ModelScope 使用的缓存目录与 HuggingFace 不同。您可能需要手动设置以避免磁盘空间不足。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/utils.py` | `SGLANG_USE_MODELSCOPE` 环境变量检测：将模型下载从 HuggingFace Hub 切换到 ModelScope SDK |
| `python/sglang/srt/model_loader/loader.py` | 模型权重加载：在 ModelScope 模式下使用 ModelScope 的 `snapshot_download()` |
| `python/sglang/srt/server_args.py` | 服务参数处理：当 `SGLANG_USE_MODELSCOPE=true` 时通过 ModelScope 解析模型路径 |

### 集成要点

- **启用方式**：启动前执行 `export SGLANG_USE_MODELSCOPE=true`，后续所有模型下载将使用 ModelScope
- **缓存目录**：ModelScope 默认缓存在 `~/.cache/modelscope`（而非 `~/.cache/huggingface`）
- **Docker 使用**：挂载 `-v ~/.cache/modelscope:/root/.cache/modelscope` 并传入 `--env "SGLANG_USE_MODELSCOPE=true"`
