# Ollama-Compatible API

SGLang provides Ollama API compatibility, allowing you to use the Ollama CLI and Python library with SGLang as the inference backend.

**中文对照**：SGLang 提供 Ollama API 兼容性，允许你使用 Ollama CLI 和 Python 库，以 SGLang 作为推理后端。

## Prerequisites

**中文对照**：先决条件

```bash
# Install the Ollama Python library (for Python client usage)
pip install ollama
```

> **Note**: You don't need the Ollama server installed - SGLang acts as the backend. You only need the `ollama` CLI or Python library as the client.

**中文对照**：**注意**：你不需要安装 Ollama 服务器——SGLang 作为后端。你只需要 `ollama` CLI 或 Python 库作为客户端。

## Endpoints

**中文对照**：端点

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET, HEAD | Health check for Ollama CLI |
| `/api/tags` | GET | List available models |
| `/api/chat` | POST | Chat completions (streaming & non-streaming) |
| `/api/generate` | POST | Text generation (streaming & non-streaming) |
| `/api/show` | POST | Model information |

## Quick Start

**中文对照**：快速开始

### 1. Launch SGLang Server

**中文对照**：1. 启动 SGLang 服务器

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 30001 \
    --host 0.0.0.0
```

> **Note**: The model name used with `ollama run` must match exactly what you passed to `--model`.

**中文对照**：**注意**：`ollama run` 使用的模型名称必须与传递给 `--model` 的名称完全匹配。

### 2. Use Ollama CLI

**中文对照**：2. 使用 Ollama CLI

```bash
# List available models
OLLAMA_HOST=http://localhost:30001 ollama list

# Interactive chat
OLLAMA_HOST=http://localhost:30001 ollama run "Qwen/Qwen2.5-1.5B-Instruct"
```

If connecting to a remote server behind a firewall:

**中文对照**：如果连接到防火墙后面的远程服务器：

```bash
# SSH tunnel
ssh -L 30001:localhost:30001 user@gpu-server -N &

# Then use Ollama CLI as above
OLLAMA_HOST=http://localhost:30001 ollama list
```

### 3. Use Ollama Python Library

**中文对照**：3. 使用 Ollama Python 库

```python
import ollama

client = ollama.Client(host='http://localhost:30001')

# Non-streaming
response = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response['message']['content'])

# Streaming
stream = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

## Smart Router

**中文对照**：智能路由器

For intelligent routing between local Ollama (fast) and remote SGLang (powerful) using an LLM judge, see the [Smart Router documentation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/ollama/README.md).

**中文对照**：对于使用 LLM 判断器在本地 Ollama（快速）和远程 SGLang（强大）之间进行智能路由，请参阅[智能路由器文档](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/ollama/README.md)。

## Summary

**中文对照**：摘要

| Component | Purpose |
|-----------|---------|
| **Ollama API** | Familiar CLI/API that developers already know |
| **SGLang Backend** | High-performance inference engine |
| **Smart Router** | Intelligent routing - fast local for simple tasks, powerful remote for complex tasks |

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/entrypoints/ollama/` | Ollama API 兼容层：处理 `/api/tags`、`/api/chat`、`/api/generate`、`/api/show` 端点 |
| `python/sglang/srt/entrypoints/http_server.py` | 注册 Ollama 路由，与 OpenAI 兼容端点并存 |
| `python/sglang/srt/managers/tokenizer_manager.py` | 以与 OpenAI 请求相同的方式处理 Ollama chat/generate 请求 |

### 关键代码逻辑

- **端点映射**：Ollama `/api/chat` -> 内部转换为 OpenAI `/v1/chat/completions` 格式 -> 由相同的分词器/调度器管道处理
- **模型名称匹配**：`OLLAMA_HOST` 必须指向 SGLang 服务器；`ollama run` 中的模型名称必须与 `--model` 参数完全匹配
- **智能路由器**：位于 `python/sglang/srt/entrypoints/ollama/`，使用 LLM 判断器在本地 Ollama 和远程 SGLang 之间路由

### 集成要点

- **服务器启动**：标准启动命令 `python -m sglang.launch_server --model ...` — Ollama 端点自动可用
- **客户端**：使用 `ollama` CLI 或 Python 库作为客户端，将 `OLLAMA_HOST` 指向 SGLang 服务器 URL
