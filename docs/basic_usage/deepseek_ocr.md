# DeepSeek OCR (OCR-1 / OCR-2)

DeepSeek OCR models are multimodal (image + text) models for OCR and document understanding.

**中文对照**：DeepSeek OCR 模型是多模态（图像+文本）模型，用于光学字符识别和文档理解。

## Launch server

**中文对照**：启动服务器

```shell
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-OCR-2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000
```

> You can replace `deepseek-ai/DeepSeek-OCR-2` with `deepseek-ai/DeepSeek-OCR`.

**中文对照**：你可以将 `deepseek-ai/DeepSeek-OCR-2` 替换为 `deepseek-ai/DeepSeek-OCR`。

## Prompt examples

**中文对照**：提示词示例

Recommended prompts from the model card:

**中文对照**：来自模型卡的推荐提示词：

```
<image>
<|grounding|>Convert the document to markdown.
```

```
<image>
Free OCR.
```

## OpenAI-compatible request example

**中文对照**：OpenAI 兼容请求示例

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>\n<|grounding|>Convert the document to markdown."},
                {"type": "image_url", "image_url": {"url": "https://example.com/your_image.jpg"}},
            ],
        }
    ],
    "max_tokens": 512,
}

response = requests.post(url, json=data)
print(response.text)
```

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/deepseek_vl_v2.py` | DeepSeek OCR/VL 模型架构，包含视觉编码器 + 语言解码器 |
| `python/sglang/srt/multimodal/mm_utils.py` | 图像预处理：加载、缩放、格式转换以支持 OCR 输入 |
| `python/sglang/srt/entrypoints/http_server.py` | `/v1/chat/completions` 端点，处理多模态（图像 + 文本）请求 |

### 集成要点

- **服务器启动**：标准启动命令 `python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-OCR-2 --trust-remote-code`
- **API**：兼容 OpenAI 的 `/v1/chat/completions` 端点，支持 `image_url` 内容类型
- **提示词**：使用 `<image>` 标签 + 定位标记（如 `<|grounding|>`）进行 OCR 任务
