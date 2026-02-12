# Embedding Models

SGLang provides robust support for embedding models by integrating efficient serving mechanisms with its flexible programming interface. This integration allows for streamlined handling of embedding tasks, facilitating faster and more accurate retrieval and semantic search operations. SGLang's architecture enables better resource utilization and reduced latency in embedding model deployment.

**中文对照**：嵌入模型

SGLang 通过将高效的服务机制与其灵活的编程接口集成，为嵌入模型提供了强大的支持。这种集成允许简化嵌入任务的处理，促进更快、更准确的检索和语义搜索操作。SGLang 的架构实现了更好的资源利用率和降低的嵌入模型部署延迟。

```{important}
Embedding models are executed with `--is-embedding` flag and some may require `--trust-remote-code`
```

## Quick Start

### Launch Server

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-4B \
  --is-embedding \
  --host 0.0.0.0 \
  --port 30000
```

### Client Request

```python
import requests

url = "http://127.0.0.1:30000"

payload = {
    "model": "Qwen/Qwen3-Embedding-4B",
    "input": "What is the capital of France?",
    "encoding_format": "float"
}

response = requests.post(url + "/v1/embeddings", json=payload).json()
print("Embedding:", response["data"][0]["embedding"])
```

**中文对照**：快速入门

### 启动服务器

**中文对照**：启动服务器

### 客户端请求

**中文对照**：客户端请求

## Multimodal Embedding Example

For multimodal models like GME that support both text and images:

```shell
python3 -m sglang.launch_server \
  --model-path Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
  --is-embedding \
  --chat-template gme-qwen2-vl \
  --host 0.0.0.0 \
  --port 30000
```

```python
import requests

url = "http://127.0.0.1:30000"

text_input = "Represent this image in embedding space."
image_path = "https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/images/023.jpg"

payload = {
    "model": "gme-qwen2-vl",
    "input": [
        {
            "text": text_input
        },
        {
            "image": image_path
        }
    ],
}

response = requests.post(url + "/v1/embeddings", json=payload).json()

print("Embeddings:", [x.get("embedding") for x in response.get("data", [])])
```

**中文对照**：多模态嵌入示例

对于支持文本和图像的多模态模型（如 GME）：

## Matryoshka Embedding Example

[Matryoshka Embeddings](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings) or [Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) is a technique used in training embedding models. It allows user to trade off between performance and cost.

### 1. Launch a Matryoshka‑capable model

If the model config already includes `matryoshka_dimensions` or `is_matryoshka` then no override is needed. Otherwise, you can use `--json-model-override-args` as below:

```shell
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-Embedding-0.6B \
    --is-embedding \
    --host 0.0.0.0 \
    --port 30000 \
    --json-model-override-args '{"matryoshka_dimensions": [128, 256, 512, 1024, 1536]}'
```

1. Setting `"is_matryoshka": true` allows truncating to any dimension. Otherwise, the server will validate that the specified dimension in the request is one of `matryoshka_dimensions`.
2. Omitting `dimensions` in a request returns the full vector.

### 2. Make requests with different output dimensions

```python
import requests

url = "http://127.0.0.1:30000"

# Request a truncated (Matryoshka) embedding by specifying a supported dimension.
payload = {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Explain diffusion models simply.",
    "dimensions": 512  # change to 128 / 1024 / omit for full size
}

response = requests.post(url + "/v1/embeddings", json=payload).json()
print("Embedding:", response["data"][0]["embedding"])
```

**中文对照**：Matryoshka 嵌入示例

[Matryoshka 嵌入](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings) 或 [Matryoshka 表示学习（MRL）](https://arxiv.org/abs/2205.13147) 是用于训练嵌入模型的技术。它允许用户在性能和成本之间进行权衡。

### 1. 启动支持 Matryoshka 的模型

如果模型配置已包含 `matryoshka_dimensions` 或 `is_matryoshka`，则不需要覆盖。否则，您可以使用 `--json-model-override-args` 如下：

**中文对照**：1. 启动支持 Matryoshka 的模型

1. 设置 `"is_matryoshka": true` 允许截断到任何维度。否则，服务器将验证请求中指定的维度是否为 `matryoshka_dimensions` 之一。
2. 在请求中省略 `dimensions` 返回完整向量。

**中文对照**：1. 设置 `"is_matryoshka": true` 允许截断到任何维度。否则，服务器将验证请求中指定的维度是否为 `matryoshka_dimensions` 之一。
2. 在请求中省略 `dimensions` 返回完整向量。

### 2. 使用不同的输出维度发出请求

**中文对照**：2. 使用不同的输出维度发出请求

## Supported Models

| Model Family                               | Example Model                          | Chat Template | Description                                                                 |
| ------------------------------------------ | -------------------------------------- | ------------- | --------------------------------------------------------------------------- |
| **E5 (Llama/Mistral based)**              | `intfloat/e5-mistral-7b-instruct`     | N/A           | High-quality text embeddings based on Mistral/Llama architectures          |
| **GTE-Qwen2**                             | `Alibaba-NLP/gte-Qwen2-7B-instruct`   | N/A           | Alibaba's text embedding model with multilingual support                   |
| **Qwen3-Embedding**                       | `Qwen/Qwen3-Embedding-4B`             | N/A           | Latest Qwen3-based text embedding model for semantic representation        |
| **BGE**                                    | `BAAI/bge-large-en-v1.5`              | N/A           | BAAI's text embeddings (requires `attention-backend` triton/torch_native)  |
| **GME (Multimodal)**                      | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`| `gme-qwen2-vl`| Multimodal embedding for text and image cross-modal tasks                  |
| **CLIP**                                   | `openai/clip-vit-large-patch14-336`   | N/A           | OpenAI's CLIP for image and text embeddings                                |

**中文对照**：支持的模型

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/entrypoints/http_server.py` | `/v1/embeddings` API 端点：处理嵌入请求并返回 embedding 向量 |
| `python/sglang/srt/server_args.py` | `--is-embedding` 标志：启用嵌入模型运行模式 |
| `python/sglang/srt/models/` | 嵌入模型实现：如 E5、GTE、BGE 等基于 Llama/Mistral/Qwen 架构的嵌入模型 |
| `python/sglang/srt/configs/model_config.py` | 嵌入模型配置：检测 `matryoshka_dimensions` 和 `is_matryoshka` 参数 |

### 集成要点

- **启动方式**：所有嵌入模型需要 `--is-embedding` 标志，部分模型还需 `--trust-remote-code`
- **多模态嵌入**：GME 等模型支持文本和图像的跨模态嵌入，需指定 `--chat-template`
- **Matryoshka 嵌入**：支持可变维度输出，通过请求中的 `dimensions` 参数截断到指定维度
- **注意力后端**：BGE 模型需要使用 `--attention-backend triton` 或 `torch_native`
