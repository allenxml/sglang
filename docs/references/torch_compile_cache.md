# Enabling cache for torch.compile

SGLang uses `max-autotune-no-cudagraphs` mode of torch.compile. The auto-tuning can be slow.
If you want to deploy a model on many different machines, you can ship the torch.compile cache to these machines and skip the compilation steps.

**中文对照**：SGLang 使用 torch.compile 的 `max-autotune-no-cudagraphs` 模式。自动调优可能很慢。
如果您想在许多不同的机器上部署模型，您可以将这些机器的 torch.compile 缓存发送过去并跳过编译步骤。

This is based on https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html

**中文对照**：这基于 https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html

1. Generate the cache by setting TORCHINDUCTOR_CACHE_DIR and running the model once.
```
TORCHINDUCTOR_CACHE_DIR=/root/inductor_root_cache python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --enable-torch-compile
```
2. Copy the cache folder to other machines and launch the server with `TORCHINDUCTOR_CACHE_DIR`.

**中文对照**：
TORCHINDUCTOR_CACHE_DIR=/root/inductor_root_cache python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --enable-torch-compile
```
2. 将缓存文件夹复制到其他机器并使用 `TORCHINDUCTOR_CACHE_DIR` 启动服务器。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/server_args.py` | `--enable-torch-compile` 命令行参数定义 |
| `python/sglang/srt/utils/common.py` | `maybe_torch_compile()`：条件式 torch.compile 包装器，在模型各层中使用 |
| `python/sglang/srt/model_executor/cuda_graph_runner.py` | `set_torch_compile_config()`：配置 `max-autotune-no-cudagraphs` 编译模式 |
| `python/sglang/srt/utils/patch_torch.py` | `monkey_patch_torch_compile()`：为 SGLang 兼容性打补丁 |

### 集成要点

- **缓存机制**：环境变量 `TORCHINDUCTOR_CACHE_DIR` 控制 torch.compile 编译后内核缓存的存储位置
- **编译模式**：使用 `max-autotune-no-cudagraphs` — 自动调优内核但由 SGLang 自行管理 CUDA Graph
- **运行时控制**：环境变量 `SGLANG_ENABLE_TORCH_COMPILE`（默认 `true`）控制 torch.compile 是否在运行时可用
