# Custom Chat Template

**NOTE**: There are two chat template systems in SGLang project. This document is about setting a custom chat template for the OpenAI-compatible API server (defined at [conversation.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/conversation.py)). It is NOT related to the chat template used in the SGLang language frontend (defined at [chat_template.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/chat_template.py)).

**中文对照**：SGLang 项目中有两套聊天模板系统。本文档是关于为 OpenAI 兼容的 API 服务器设置自定义聊天模板（定义在 [conversation.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/conversation.py) 中）。它与 SGLang 语言前端使用的聊天模板无关（定义在 [chat_template.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/chat_template.py) 中）。

By default, the server uses the chat template specified in the model tokenizer from Hugging Face.
It should just work for most official models such as Llama-2/Llama-3.

**中文对照**：默认情况下，服务器使用 Hugging Face 模型分词器中指定的聊天模板。对于大多数官方模型（如 Llama-2/Llama-3），它应该可以直接工作。

If needed, you can also override the chat template when launching the server:

**中文对照**：如果需要，您也可以在启动服务器时覆盖聊天模板：

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-chat-hf \
  --port 30000 \
  --chat-template llama-2
```

If the chat template you are looking for is missing, you are welcome to contribute it or load it from a file.

**中文对照**：如果您正在寻找的聊天模板缺失，欢迎您贡献它或从文件加载。

## JSON Format

You can load the JSON format, which is defined by `conversation.py`.

**中文对照**：您可以加载由 `conversation.py` 定义的 JSON 格式。

```json
{
  "name": "my_model",
  "system": "<|im_start|>system",
  "user": "<|im_start|>user",
  "assistant": "<|im_start|>assistant",
  "sep_style": "CHATML",
  "sep": "<|im_end|>",
  "stop_str": ["<|im_end|>", "<|im_start|>"]
}
```

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-chat-hf \
  --port 30000 \
  --chat-template ./my_model_template.json
```

## Jinja Format

You can also use the [Jinja template format](https://huggingface.co/docs/transformers/main/en/chat_templating) as defined by Hugging Face Transformers.

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-chat-hf \
  --port 30000 \
  --chat-template ./my_model_template.jinja
```

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/conversation.py` | 聊天模板注册表：内置模板（Llama-2、ChatML 等）和 JSON 格式解析器 |
| `python/sglang/srt/server_args.py` | `--chat-template` 和 `--completion-template` CLI 参数 |
| `python/sglang/srt/managers/tokenizer_manager.py` | 在分词期间应用聊天模板；从模型或 CLI 覆盖解析模板 |
| `python/sglang/lang/chat_template.py` | SGLang 前端聊天模板（独立于服务器端的 conversation.py） |

### 集成要点

- **模板解析顺序**：HuggingFace 分词器默认 → `--chat-template` CLI 覆盖（内置名称、JSON 文件或 Jinja 文件）
- **JSON 格式**：由 `conversation.py` 解析；字段：`name`、`system`、`user`、`assistant`、`sep_style`、`sep`、`stop_str`
- **Jinja 格式**：直接作为 HuggingFace 兼容的 Jinja2 模板字符串加载
