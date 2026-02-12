# Sampling Parameters

This doc describes the sampling parameters of the SGLang Runtime. It is the low-level endpoint of the runtime.
If you want a high-level endpoint that can automatically handle chat templates, consider using the [OpenAI Compatible API](openai_api_completions.ipynb).

**中文对照**：本文档描述了 SGLang 运行时的采样参数。它是运行时的低级别端点。
如果你想要可以自动处理聊天模板的高级端点，请考虑使用 [OpenAI 兼容 API](openai_api_completions.ipynb)。

## `/generate` Endpoint

**中文对照**：`/generate` 端点

The `/generate` endpoint accepts the following parameters in JSON format. For detailed usage, see the [native API doc](native_api.ipynb). The object is defined at `io_struct.py::GenerateReqInput`. You can also read the source code to find more arguments and docs.

**中文对照**：`/generate` 端点接受 JSON 格式的以下参数。详细用法请参阅[本机 API 文档](native_api.ipynb)。该对象在 `io_struct.py::GenerateReqInput` 中定义。你也可以阅读源代码以查找更多参数和文档。

| Argument                   | Type/Default                                                                 | Description |
|----------------------------|------------------------------------------------------------------------------|-------------|
| text                       | `Optional[Union[List[str], str]] = None`                                     | The input prompt. Can be a single prompt or a batch of prompts. |
| input_ids                  | `Optional[Union[List[List[int]], List[int]]] = None`                         | The token IDs for text; one can specify either text or input_ids. |
| input_embeds               | `Optional[Union[List[List[List[float]]], List[List[float]]]] = None`         | The embeddings for input_ids; one can specify either text, input_ids, or input_embeds. |
| image_data                 | `Optional[Union[List[List[ImageDataItem]], List[ImageDataItem], ImageDataItem]] = None` | The image input. Supports three formats: (1) **Raw images**: PIL Image, file path, URL, or base64 string; (2) **Processor output**: Dict with `format: "processor_output"` containing HuggingFace processor outputs; (3) **Precomputed embeddings**: Dict with `format: "precomputed_embedding"` and `feature` containing pre-calculated visual embeddings. Can be a single image, list of images, or list of lists of images. See [Multimodal Input Formats](#multimodal-input-formats) for details. |
| audio_data                 | `Optional[Union[List[AudioDataItem], AudioDataItem]] = None`                 | The audio input. Can be a file name, URL, or base64 encoded string. |
| sampling_params            | `Optional[Union[List[Dict], Dict]] = None`                                   | The sampling parameters as described in the sections below. |
| rid                        | `Optional[Union[List[str], str]] = None`                                     | The request ID. |
| return_logprob             | `Optional[Union[List[bool], bool]] = None`                                   | Whether to return log probabilities for tokens. |
| logprob_start_len          | `Optional[Union[List[int], int]] = None`                                     | If return_logprob, the start location in the prompt for returning logprobs. Default is "-1", which returns logprobs for output tokens only. |
| top_logprobs_num           | `Optional[Union[List[int], int]] = None`                                     | If return_logprob, the number of top logprobs to return at each position. |
| token_ids_logprob          | `Optional[Union[List[List[int]], List[int]]] = None`                         | If return_logprob, the token IDs to return logprob for. |
| return_text_in_logprobs    | `bool = False`                                                               | Whether to detokenize tokens in text in the returned logprobs. |
| stream                     | `bool = False`                                                               | Whether to stream output. |
| lora_path                  | `Optional[Union[List[Optional[str]], Optional[str]]] = None`                 | The path to the LoRA. |
| custom_logit_processor     | `Optional[Union[List[Optional[str]], str]] = None`                           | Custom logit processor for advanced sampling control. Must be a serialized instance of `CustomLogitProcessor` using its `to_str()` method. For usage see below. |
| return_hidden_states       | `Union[List[bool], bool] = False`                                            | Whether to return hidden states. |
| return_routed_experts      | `bool = False`                                                               | Whether to return routed experts for MoE models. Requires `--enable-return-routed-experts` server flag. Returns base64-encoded int32 expert IDs as a flattened array with logical shape `[num_tokens, num_layers, top_k]`. |

## Sampling parameters

**中文对照**：采样参数

The object is defined at `sampling_params.py::SamplingParams`. You can also read the source code to find more arguments and docs.

**中文对照**：该对象在 `sampling_params.py::SamplingParams` 中定义。你也可以阅读源代码以查找更多参数和文档。

### Note on defaults

**中文对照**：关于默认值的说明

By default, SGLang initializes several sampling parameters from the model's `generation_config.json` (when the server is launched with `--sampling-defaults model`, which is the default). To use SGLang/OpenAI constant defaults instead, start the server with `--sampling-defaults openai`. You can always override any parameter per request via `sampling_params`.

**中文对照**：默认情况下，SGLang 从模型的 `generation_config.json` 初始化几个采样参数（当服务器使用 `--sampling-defaults model` 启动时，这是默认值）。要使用 SGLang/OpenAI 常量默认值改而使用 `--sampling-defaults openai` 启动服务器。你始终可以通过 `sampling_params` 在每个请求中覆盖任何参数。

```bash
# Use model-provided defaults from generation_config.json (default behavior)
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults model

# Use SGLang/OpenAI constant defaults instead
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults openai
```

### Core parameters

**中文对照**：核心参数

| Argument        | Type/Default                                 | Description |
|-----------------|----------------------------------------------|-------------|
| max_new_tokens  | `int = 128`                                  | The maximum output length measured in tokens. |
| stop            | `Optional[Union[str, List[str]]] = None`     | One or multiple [stop words](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop). Generation will stop if one of these words is sampled. |
| stop_token_ids  | `Optional[List[int]] = None`                 | Provide stop words in the form of token IDs. Generation will stop if one of these token IDs is sampled. |
| stop_regex      | `Optional[Union[str, List[str]]] = None`     | Stop when hitting any of the regex patterns in this list |
| temperature     | `float (model default; fallback 1.0)`        | [Temperature](https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature) when sampling the next token. `temperature = 0` corresponds to greedy sampling, a higher temperature leads to more diversity. |
| top_p           | `float (model default; fallback 1.0)`        | [Top-p](https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p) selects tokens from the smallest sorted set whose cumulative probability exceeds `top_p`. When `top_p = 1`, this reduces to unrestricted sampling from all tokens. |
| top_k           | `int (model default; fallback -1)`           | [Top-k](https://developer.nvidia.com/blog/how-to-get-better-outputs-from-your-large-language-model/#predictability_vs_creativity) randomly selects from the `k` highest-probability tokens. |
| min_p           | `float (model default; fallback 0.0)`        | [Min-p](https://github.com/huggingface/transformers/issues/27670) samples from tokens with probability larger than `min_p * highest_token_probability`. |

### Penalizers

**中文对照**：惩罚因子

| Argument           | Type/Default           | Description |
|--------------------|------------------------|-------------|
| frequency_penalty  | `float = 0.0`          | Penalizes tokens based on their frequency in generation so far. Must be between `-2` and `2` where negative numbers encourage repeatment of tokens and positive number encourages sampling of new tokens. The scaling of penalization grows linearly with each appearance of a token. |
| presence_penalty   | `float = 0.0`          | Penalizes tokens if they appeared in the generation so far. Must be between `-2` and `2` where negative numbers encourage repeatment of tokens and positive number encourages sampling of new tokens. The scaling of the penalization is constant if a token occurred. |
| repetition_penalty | `float = 1.0`          | Scales the logits of previously generated tokens to discourage (values > 1) or encourage (values < 1) repetition. Valid range is `[0, 2]`; `1.0` leaves probabilities unchanged. |
| min_new_tokens     | `int = 0`              | Forces the model to generate at least `min_new_tokens` until a stop word or EOS token is sampled. Note that this might lead to unintended behavior, for example, if the distribution is highly skewed towards these tokens. |

### Constrained decoding

**中文对照**：约束解码

Please refer to our dedicated guide on [constrained decoding](../advanced_features/structured_outputs.ipynb) for the following parameters.

**中文对照**：请参阅我们关于[约束解码](../advanced_features/structured_outputs.ipynb)的专用指南，了解以下参数。

| Argument        | Type/Default                    | Description |
|-----------------|---------------------------------|-------------|
| json_schema     | `Optional[str] = None`          | JSON schema for structured outputs. |
| regex           | `Optional[str] = None`          | Regex for structured outputs. |
| ebnf            | `Optional[str] = None`          | EBNF for structured outputs. |
| structural_tag  | `Optional[str] = None`          | The structural tag for structured outputs. |

### Other options

**中文对照**：其他选项

| Argument                      | Type/Default                    | Description |
|-------------------------------|---------------------------------|-------------|
| n                             | `int = 1`                       | Specifies the number of output sequences to generate per request. (Generating multiple outputs in one request (n > 1) is discouraged; repeating the same prompts several times offers better control and efficiency.) |
| ignore_eos                    | `bool = False`                  | Don't stop generation when EOS token is sampled. |
| skip_special_tokens           | `bool = True`                   | Remove special tokens during decoding. |
| spaces_between_special_tokens | `bool = True`                   | Whether or not to add spaces between special tokens during detokenization. |
| no_stop_trim                  | `bool = False`                  | Don't trim stop words or EOS token from the generated text. |
| custom_params                 | `Optional[List[Optional[Dict[str, Any]]]] = None` | Used when employing `CustomLogitProcessor`. For usage, see below. |

## Examples

**中文对照**：示例

### Normal

**中文对照**：普通

Launch a server:

**中文对照**：启动服务器：

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

Send a request:

**中文对照**：发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())
```

Detailed example in [send request](./send_request.ipynb).

**中文对照**：详细示例见[发送请求](./send_request.ipynb)。

### Streaming

**中文对照**：流式

Send a request and stream the output:

**中文对照**：发送请求并流式输出：

```python
import requests, json

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
        "stream": True,
    },
    stream=True,
)

prev = 0
for chunk in response.iter_lines(decode_unicode=False):
    chunk = chunk.decode("utf-8")
    if chunk and chunk.startswith("data:"):
        if chunk == "data: [DONE]":
            break
        data = json.loads(chunk[5:].strip("\n"))
        output = data["text"].strip()
        print(output[prev:], end="", flush=True)
        prev = len(output)
print("")
```

Detailed example in [openai compatible api](openai_api_completions.ipynb).

**中文对照**：详细示例见 [OpenAI 兼容 API](openai_api_completions.ipynb)。

### Multimodal

**中文对照**：多模态

Launch a server:

**中文对照**：启动服务器：

```bash
python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov
```

Download an image:

**中文对照**：下载图像：

```bash
curl -o example_image.png -L https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true
```

Send a request:

**中文对照**：发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<image>\nDescribe this image in a very short sentence.<|im_end|>\n"
                "<|im_start|>assistant\n",
        "image_data": "example_image.png",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())
```

The `image_data` can be a file name, a URL, or a base64 encoded string. See also `python/sglang/srt/utils.py:load_image`.

**中文对照**：`image_data` 可以是文件名、URL 或 base64 编码字符串。另见 `python/sglang/srt/utils.py:load_image`。

Streaming is supported in a similar manner as [above](#streaming).

**中文对照**：流式支持与[上文](#streaming)类似。

Detailed example in [OpenAI API Vision](openai_api_vision.ipynb).

**中文对照**：详细示例见 [OpenAI API Vision](openai_api_vision.ipynb)。

### Structured Outputs (JSON, Regex, EBNF)

**中文对照**：结构化输出（JSON、Regex、EBNF）

You can specify a JSON schema, regular expression or [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) to constrain the model output. The model output will be guaranteed to follow the given constraints. Only one constraint parameter (`json_schema`, `regex`, or `ebnf`) can be specified for a request.

**中文对照**：你可以指定 JSON 模式、正则表达式或 [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) 来约束模型输出。模型输出将保证遵循给定的约束。每个请求只能指定一个约束参数（`json_schema`、`regex` 或 `ebnf`）。

SGLang supports two grammar backends:

**中文对照**：SGLang 支持两个语法后端：

- [XGrammar](https://github.com/mlc-ai/xgrammar) (default): Supports JSON schema, regular expression, and EBNF constraints.
  - XGrammar currently uses the [GGML BNF format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md).
- [Outlines](https://github.com/dottxt-ai/outlines): Supports JSON schema and regular expression constraints.

**中文对照**：- [XGrammar](https://github.com/mlc-ai/xgrammar)（默认）：支持 JSON 模式、正则表达式和 EBNF 约束。
  - XGrammar 目前使用 [GGML BNF 格式](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)。
- [Outlines](https://github.com/dottxt-ai/outlines)：支持 JSON 模式和正则表达式约束。

If instead you want to initialize the Outlines backend, you can use `--grammar-backend outlines` flag:

**中文对照**：如果想要初始化 Outlines 后端，可以使用 `--grammar-backend outlines` 标志：

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--port 30000 --host 0.0.0.0 --grammar-backend [xgrammar|outlines] # xgrammar or outlines (default: xgrammar)
```

```python
import json
import requests

json_schema = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": "^[\\w]+$"},
        "population": {"type": "integer"},
    },
    "required": ["name", "population"],
})

# JSON (works with both Outlines and XGrammar)
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Here is the information of the capital of France in the JSON format.\n",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json_schema,
        },
    },
)
print(response.json())

# Regular expression (Outlines backend only)
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Paris is the capital of",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "regex": "(France|England)",
        },
    },
)
print(response.json())

# EBNF (XGrammar backend only)
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Write a greeting.",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "ebnf": 'root ::= "Hello" | "Hi" | "Hey"',
        },
    },
)
print(response.json())
```

Detailed example in [structured outputs](../advanced_features/structured_outputs.ipynb).

**中文对照**：详细示例见[结构化输出](../advanced_features/structured_outputs.ipynb)。

### Custom logit processor

**中文对照**：自定义 logit 处理器

Launch a server with `--enable-custom-logit-processor` flag on.

**中文对照**：使用 `--enable-custom-logit-processor` 标志启动服务器。

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --port 30000 \
  --enable-custom-logit-processor
```

Define a custom logit processor that will always sample a specific token id.

**中文对照**：定义一个自定义 logit 处理器，它将始终采样特定的令牌 ID。

```python
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

class DeterministicLogitProcessor(CustomLogitProcessor):
    """A dummy logit processor that changes the logits to always
    sample the given token id.
    """

    def __call__(self, logits, custom_param_list):
        # Check that the number of logits matches the number of custom parameters
        assert logits.shape[0] == len(custom_param_list)
        key = "token_id"

        for i, param_dict in enumerate(custom_param_list):
            # Mask all other tokens
            logits[i, :] = -float("inf")
            # Assign highest probability to the specified token
            logits[i, param_dict[key]] = 0.0
        return logits
```

Send a request:

**中文对照**：发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 32,
            "custom_params": {"token_id": 5},
        },
    },
)
print(response.json())
```

Send an OpenAI chat completion request:

**中文对照**：发送 OpenAI 聊天完成请求：

```python
import openai
from sglang.utils import print_highlight

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0.0,
    max_tokens=32,
    extra_body={
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "custom_params": {"token_id": 5},
    },
)

print_highlight(f"Response: {response}")
```

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/sampling/sampling_params.py` | `SamplingParams` 数据类：所有采样参数的定义、验证和默认值 |
| `python/sglang/srt/layers/sampler.py` | `Sampler.forward()`：应用 temperature、top-p、top-k、min-p、惩罚，然后从 logits 中采样令牌 ID |
| `python/sglang/srt/managers/io_struct.py` | `GenerateReqInput` 请求模式：text/input_ids/image_data/sampling_params 字段 |
| `python/sglang/srt/entrypoints/http_server.py` | `/generate` 端点处理器：解析 JSON 请求体并分发到分词器管理器 |
| `python/sglang/srt/sampling/custom_logit_processor.py` | `CustomLogitProcessor` 基类，用于用户自定义 logit 操作 |
| `python/sglang/srt/constrained/xgrammar_backend.py` | XGrammar FSM 后端，用于结构化输出约束（JSON schema、正则表达式、EBNF） |

### 架构

```
[客户端 JSON 请求]
  sampling_params: {temperature, top_p, top_k, ...}
        |
        ▼
[HTTP 服务器] ──解析──▶ GenerateReqInput + SamplingParams
        |
        ▼
[TokenizerManager] → [Scheduler] → [ModelRunner]
        |
        ▼
[Sampler.forward()]
  1. 应用温度缩放：logits / temperature
  2. 应用惩罚：frequency_penalty、presence_penalty、repetition_penalty
  3. 应用 top-k 过滤
  4. 应用 top-p（核采样）过滤
  5. 应用 min-p 过滤
  6. 应用语法掩码（如果有 json_schema/regex/ebnf）
  7. 应用 custom_logit_processor（如果启用）
  8. 从过滤后的分布中采样令牌 ID
```

### 关键代码逻辑

- **参数默认值**：当使用 `--sampling-defaults model`（默认）时，`SamplingParams` 从 `generation_config.json` 读取默认值；使用 `--sampling-defaults openai` 时回退到 OpenAI 常量
- **约束解码**：`json_schema`、`regex`、`ebnf` 参数触发 `xgrammar_backend.py` 中的语法编译，生成在采样前应用的令牌掩码
- **自定义 logit 处理器**：通过 `to_str()` 序列化，在服务器端反序列化；在 GPU logits 张量上运行，每个请求带有 `custom_params` 字典

### 集成要点

- **原生 API**：`POST /generate` 带有 `sampling_params` 字典
- **OpenAI API**：`POST /v1/chat/completions` 将 OpenAI 参数映射到 `SamplingParams`
- **服务器参数**：`--sampling-defaults model|openai`、`--grammar-backend xgrammar|outlines`、`--enable-custom-logit-processor`
