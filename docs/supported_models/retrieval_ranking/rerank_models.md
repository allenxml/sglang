# Rerank Models

SGLang offers comprehensive support for rerank models by incorporating optimized serving frameworks with a flexible programming interface. This setup enables efficient processing of cross-encoder reranking tasks, improving the accuracy and relevance of search result ordering. SGLang's design ensures high throughput and low latency during reranker model deployment, making it ideal for semantic-based result refinement in large-scale retrieval systems.

**中文对照**：重排序模型

SGLang 通过结合优化的服务框架和灵活的编程接口，为重排序模型提供了全面的支持。这种设置能够高效处理交叉编码器重排序任务，提高搜索结果排序的准确性和相关性。SGLang 的设计确保了在重排序模型部署期间的高吞吐量和低延迟，使其成为大规模检索系统中基于语义的结果细化的理想选择。

```{important}
Rerank models in SGLang fall into two categories:

- **Cross-encoder rerank models**: run with `--is-embedding` (embedding runner).
- **Decoder-only rerank models**: run **without** `--is-embedding` and use next-token logprob scoring (yes/no).
  - Text-only (e.g. Qwen3-Reranker)
  - Multimodal (e.g. Qwen3-VL-Reranker): also supports image/video content

Some models may require `--trust-remote-code`.
```

## Supported rerank models

| Model Family (Rerank)                          | Example HuggingFace Identifier       | Chat Template | Description                                                                                                                      |
|------------------------------------------------|--------------------------------------|---------------|----------------------------------------------------------------------------------------------------------------------------------|
| **BGE-Reranker (BgeRerankModel)**              | `BAAI/bge-reranker-v2-m3`            | N/A           | Currently only support `attention-backend` `triton` and `torch_native`. High-performance cross-encoder reranker model from BAAI. Suitable for reranking search results based on semantic relevance.   |
| **Qwen3-Reranker (decoder-only yes/no)**       | `Qwen/Qwen3-Reranker-8B`             | `examples/chat_template/qwen3_reranker.jinja` | Decoder-only reranker using next-token logprob scoring for labels (yes/no). Launch **without** `--is-embedding`. |
| **Qwen3-VL-Reranker (multimodal yes/no)**      | `Qwen/Qwen3-VL-Reranker-2B`          | `examples/chat_template/qwen3_vl_reranker.jinja` | Multimodal decoder-only reranker supporting text, images, and videos. Uses yes/no logprob scoring. Launch **without** `--is-embedding`. |


## Cross-Encoder Rerank (embedding runner)

**中文对照**：交叉编码器重排序（嵌入运行器）

### Launch Command

**中文对照**：启动命令

```shell
python3 -m sglang.launch_server \
  --model-path BAAI/bge-reranker-v2-m3 \
  --host 0.0.0.0 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend triton \
  --is-embedding \
  --port 30000
```

### Example Client Request

**中文对照**：客户端请求示例

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "what is panda?",
    "documents": [
        "hi",
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
    ],
    "top_n": 1,
    "return_documents": True
}

response = requests.post(url, json=payload)
response_json = response.json()

for item in response_json:
    if item.get("document"):
        print(f"Score: {item['score']:.2f} - Document: '{item['document']}'")
    else:
        print(f"Score: {item['score']:.2f} - Index: {item['index']}")
```

**Request Parameters:**

- `query` (required): The query text to rank documents against
- `documents` (required): List of documents to be ranked
- `model` (required): Model to use for reranking
- `top_n` (optional): Maximum number of documents to return. Defaults to returning all documents. If specified value is greater than the total number of documents, all documents will be returned.
- `return_documents` (optional): Whether to return documents in the response. Defaults to `True`.

**中文对照**：请求参数

## Qwen3-Reranker (decoder-only yes/no rerank)

**中文对照**：Qwen3-Reranker（仅解码器 yes/no 重排序）

### Launch Command

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B \
  --trust-remote-code \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 8001 \
  --chat-template examples/chat_template/qwen3_reranker.jinja
```

```{note}
Qwen3-Reranker uses decoder-only logprob scoring (yes/no). Do NOT launch it with `--is-embedding`.
```

### Example Client Request (supports optional instruct, top_n, and return_documents)

```shell
curl -X POST http://127.0.0.1:8001/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Reranker-0.6B",
    "query": "法国首都是哪里？",
    "documents": [
      "法国的首都是巴黎。",
      "德国的首都是柏林。",
      "香蕉是黄色的水果。"
    ],
    "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
    "top_n": 2,
    "return_documents": true
  }'
```

**Request Parameters:**

- `query` (required): The query text to rank documents against
- `documents` (required): List of documents to be ranked
- `model` (required): Model to use for reranking
- `instruct` (optional): Instruction text for the reranker
- `top_n` (optional): Maximum number of documents to return. Defaults to returning all documents. If specified value is greater than the total number of documents, all documents will be returned.
- `return_documents` (optional): Whether to return documents in the response. Defaults to `True`.

**中文对照**：### 启动命令

**中文对照**：启动命令

### 客户端请求示例

**中文对照**：客户端请求示例

**请求参数：**

**中文对照**：请求参数

**中文对照**：Qwen3-Reranker（仅解码器 yes/no 重排序）

### 启动命令

**中文对照**：启动命令

### 客户端请求示例（支持可选的 instruct、top_n 和 return_documents）

**中文对照**：客户端请求示例

**请求参数：**

- `query` (required): The query text to rank documents against
- `documents` (required): List of documents to be ranked
- `model` (required): Model to use for reranking
- `instruct` (optional): Instruction text for the reranker
- `top_n` (optional): Maximum number of documents to return. Defaults to returning all documents. If specified value is greater than the total number of documents, all documents will be returned.
- `return_documents` (optional): Whether to return documents in the response. Defaults to `True`.

**中文对照**：请求参数

### Response Format

`/v1/rerank` returns a list of objects (sorted by descending score):

- `score`: float, higher means more relevant
- `document`: the original document string (only included when `return_documents` is `true`)
- `index`: the original index in the input `documents`
- `meta_info`: optional debug/usage info (may be present for some models)

The number of returned results is controlled by the `top_n` parameter. If `top_n` is not specified or is greater than the total number of documents, all documents are returned.

Example (with `return_documents: true`):

```json
[
  {"score": 0.99, "document": "法国的首都是巴黎。", "index": 0},
  {"score": 0.01, "document": "德国的首都是柏林。", "index": 1},
  {"score": 0.00, "document": "香蕉是黄色的水果。", "index": 2}
]
```

Example (with `return_documents: false`):

```json
[
  {"score": 0.99, "index": 0},
  {"score": 0.01, "index": 1},
  {"score": 0.00, "index": 2}
]
```

Example (with `top_n: 2`):

```json
[
  {"score": 0.99, "document": "法国的首都是巴黎。", "index": 0},
  {"score": 0.01, "document": "德国的首都是柏林。", "index": 1}
]
```

### Common Pitfalls

- If you launch Qwen3-Reranker with `--is-embedding`, `/v1/rerank` cannot compute yes/no logprob scores. Relaunch **without** `--is-embedding`.
- If you see a validation error like "score should be a valid number" and the backend returned a list, upgrade to a version that coerces `embedding[0]` into `score` for rerank responses.

**中文对照**：响应格式

`/v1/rerank` 返回一个对象列表（按分数降序排序）：

**中文对照**：响应格式

### 常见陷阱

- If you launch Qwen3-Reranker with `--is-embedding`, `/v1/rerank` cannot compute yes/no logprob scores. Relaunch **without** `--is-embedding`.
- If you see a validation error like "score should be a valid number" and the backend returned a list, upgrade to a version that coerces `embedding[0]` into `score` for rerank responses.

**中文对照**：常见陷阱

## Qwen3-VL-Reranker (multimodal decoder-only rerank)

Qwen3-VL-Reranker extends the Qwen3-Reranker to support multimodal content, allowing reranking of documents containing text, images, and videos.

**中文对照**：Qwen3-VL-Reranker（多模态仅解码器重排序）

Qwen3-VL-Reranker 扩展了 Qwen3-Reranker 以支持多模态内容，允许对包含文本、图像和文档进行重排序。

### Launch Command

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-Reranker-2B \
  --trust-remote-code \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 30000 \
  --chat-template examples/chat_template/qwen3_vl_reranker.jinja
```

```{note}
Qwen3-VL-Reranker uses decoder-only logprob scoring (yes/no) like Qwen3-Reranker. Do NOT launch it with `--is-embedding`.
```

### Text-Only Reranking (backward compatible)

**中文对照**：纯文本重排序（向后兼容）

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    "model": "Qwen3-VL-Reranker-2B",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
        "The weather in Paris is usually mild with occasional rain.",
        "Deep learning is a subset of machine learning using neural networks with many layers.",
    ],
    "instruct": "Retrieve passages that answer the question.",
    "return_documents": True
}

response = requests.post(url, json=payload)
results = response.json()

for item in results:
    print(f"Score: {item['score']:.4f} - {item['document'][:60]}...")
```

### Image Reranking (text query, image/mixed documents)

**中文对照**：图像重排序（文本查询，图像/混合文档）

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    "query": "A woman playing with her dog on a beach at sunset.",
    "documents": [
        # Document 1: Text description
        "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset.",
        # Document 2: Image URL
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/beach_dog.jpeg"
                }
            }
        ],
        # Document 3: Text + Image (mixed)
        [
            {"type": "text", "text": "A joyful scene at the beach:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/beach_dog.jpeg"
                }
            }
        ]
    ],
    "instruct": "Retrieve images or text relevant to the user's query.",
    "return_documents": False
}

response = requests.post(url, json=payload)
results = response.json()

for item in results:
    print(f"Index: {item['index']}, Score: {item['score']:.4f}")
```

### Multimodal Query Reranking (query with image)

**中文对照**：多模态查询重排序（带图像的查询）

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    # Query with text and image
    "query": [
        {"type": "text", "text": "Find similar images to this:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/reference_image.jpeg"
            }
        }
    ],
    "documents": [
        "A cat sleeping on a couch.",
        "A woman and her dog enjoying the sunset at the beach.",
        "A busy city street with cars and pedestrians.",
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/similar_image.jpeg"
                }
            }
        ]
    ],
    "instruct": "Find images or descriptions similar to the query image."
}

response = requests.post(url, json=payload)
results = response.json()

for item in results:
    print(f"Index: {item['index']}, Score: {item['score']:.4f}")
```

### Request Parameters (Multimodal)

- `query` (required): Can be a string (text-only) or a list of content parts:
  - `{"type": "text", "text": "..."}` for text
  - `{"type": "image_url", "image_url": {"url": "..."}}` for images
  - `{"type": "video_url", "video_url": {"url": "..."}}` for videos
- `documents` (required): List where each document can be a string or list of content parts (same format as query)
- `instruct` (optional): Instruction text for the reranker
- `top_n` (optional): Maximum number of documents to return
- `return_documents` (optional): Whether to return documents in the response (default: `false`)

### Common Pitfalls

- Always use `--chat-template examples/chat_template/qwen3_vl_reranker.jinja` for Qwen3-VL-Reranker.
- Do NOT launch with `--is-embedding`.
- For best results, use `--disable-radix-cache` to avoid caching issues with multimodal content.
- **Note**: Currently only `Qwen3-VL-Reranker-2B` is tested and supported. The 8B model may have different behavior and is not guaranteed to work with this template.

**中文对照**：### 请求参数（多模态）

**中文对照**：请求参数

### 常见陷阱

**中文对照**：常见陷阱

**中文对照**：### 启动命令

**中文对照**：启动命令

### 纯文本重排序（向后兼容）

**中文对照**：纯文本重排序

### 图像重排序（文本查询，图像/混合文档）

**中文对照**：图像重排序

### 多模态查询重排序（带图像的查询）

**中文对照**：多模态查询重排序

### 请求参数（多模态）

**中文对照**：请求参数

### 常见陷阱

- Always use `--chat-template examples/chat_template/qwen3_vl_reranker.jinja` for Qwen3-VL-Reranker.
- Do NOT launch with `--is-embedding`.
- For best results, use `--disable-radix-cache` to avoid caching issues with multimodal content.
- **Note**: Currently only `Qwen3-VL-Reranker-2B` is tested and supported. The 8B model may have different behavior and is not guaranteed to work with this template.

**中文对照**：常见陷阱

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/entrypoints/http_server.py` | `/v1/rerank` API 端点：处理重排序请求 |
| `python/sglang/srt/server_args.py` | `--is-embedding`、`--chat-template`、`--trust-remote-code` 命令行参数 |
| `examples/chat_template/qwen3_reranker.jinja` | Qwen3-Reranker 专用对话模板：构造 yes/no 评分格式 |
| `examples/chat_template/qwen3_vl_reranker.jinja` | Qwen3-VL-Reranker 多模态重排序模板 |

### 集成要点

- **两类重排序模型**：交叉编码器（需 `--is-embedding`）和解码器式 yes/no 评分（不加 `--is-embedding`）
- **交叉编码器**：如 BGE-Reranker，使用嵌入运行器计算相关性得分
- **解码器式重排序**：如 Qwen3-Reranker，通过下一个 token 的 logprob（yes/no）进行评分
- **多模态重排序**：Qwen3-VL-Reranker 支持文本、图像和视频内容的混合重排序
