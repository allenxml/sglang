# Transformers fallback in SGLang

`sglang` can fall back to using models that are available in `transformers`. This works for most decoder-style language models and support for vision-language models is coming soon!

**中文对照**：`sglang` 可以回退到使用 `transformers` 中可用的模型。这适用于大多数解码器风格的语言模型，对视觉语言模型的支持即将推出！

## Example launch Command

By default, we will use sglang implementation if it is available. Otherwise, we will fall back to transformers one. However, you can switch the implementation by setting `--model-impl` to `transformers`.

**中文对照**：默认情况下，如果有可用的 sglang 实现，我们将使用它。否则，我们将回退到 transformers。但是，您可以通过将 `--model-impl` 设置为 `transformers` 来切换实现。

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --model-impl transformers
```

## Supported features

### Quantization

Transformers fall back has supported most of available quantization in SGLang (except GGUF). See [Quantization page](../advanced_features/quantization.md) for more information about supported quantization in SGLang.

**中文对照**：支持的特性

### 量化

Transformers 回退已支持 SGLang 中大多数可用的量化方法（GGUF 除外）。有关 SGLang 中支持的量化的更多信息，请参阅[量化页面](../advanced_features/quantization.md)。

### Remote code

This fallback also means that any model on the hub that can be used in `transformers` with `trust_remote_code=True` that correctly implements attention can be used in production!

**中文对照**：远程代码

此回退还意味着，hub 上任何可以在 `transformers` 中与 `trust_remote_code=True` 一起使用且正确实现注意力的模型都可以在生产中使用！

A model just needs the following two things:

**中文对照**：一个模型只需要具备以下两点：

```python
from transformers import PreTrainedModel
from torch import nn

class MyAttention(nn.Module):

  def forward(self, hidden_states, **kwargs): # <- kwargs are required

    ...
    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attn_output, attn_weights = attention_interface(
      self,
      query_states,
      key_states,
      value_states,
      **kwargs,
    )
    ...

class MyModel(PreTrainedModel):
  _supports_attention_backend = True
```

Here is what happens in the background:

1. The config is loaded
2. `MyModel` python class is loaded from the `auto_map`, and we check that the model `_supports_attention_backend`.
3. The `TransformersModel` backend is used. See `/srt/models/transformers`, which leverages `self.config._attn_implementation = "sglang"`, thus the need to use `ALL_ATTENTION_FUNCTIONS`.

That's it!

**中文对照**：以下是后台发生的事情：

1. 加载配置
2. 从 `auto_map` 加载 `MyModel` python 类，并检查模型是否 `_supports_attention_backend`。
3. 使用 `TransformersModel` 后端。请参阅 `/srt/models/transformers`，它利用 `self.config._attn_implementation = "sglang"`，因此需要使用 `ALL_ATTENTION_FUNCTIONS`。

就是这样！

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/transformers/` | Transformers 回退后端：用 SGLang 的注意力实现包装 HuggingFace 模型 |
| `python/sglang/srt/server_args.py` | `--model-impl` 命令行参数：`auto`（优先原生）、`sglang`（强制原生）、`transformers`（强制回退） |
| `python/sglang/srt/configs/model_config.py` | 模型实现选择：先检查原生支持，若 `--model-impl auto` 则回退到 transformers |

### 集成要点

- **触发回退**：`--model-impl transformers` 强制使用 HuggingFace 后端；`auto` 模式在有原生支持时优先使用 SGLang
- **注意力注入**：设置 `self.config._attn_implementation = "sglang"` 将注意力路由到 `ALL_ATTENTION_FUNCTIONS`
- **模型要求**：远程代码模型必须设置 `_supports_attention_backend = True` 并在 attention forward 中使用 `ALL_ATTENTION_FUNCTIONS`
- **量化支持**：支持 SGLang 的大部分量化方法，GGUF 除外
