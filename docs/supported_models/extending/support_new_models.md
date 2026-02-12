# How to Support New Models

This document explains how to add support for new language models and multimodal large language models (MLLMs) in
SGLang. It also covers how to test new models and register external implementations.

**中文对照**：本文档介绍了如何在 SGLang 中添加对新语言模型和多模态大语言模型（MLLM）的支持。还包括如何测试新模型和注册外部实现。

## How to Support a New Language Model

To support a new model in SGLang, you only need to add a single file under
the [SGLang Models Directory](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models). You can learn
from existing model implementations and create a new file for your model. For most models, you should be able to find a
similar model to start with (e.g., starting from Llama). Also refer how
to [port a Model from vLLM to SGLang](#port-a-model-from-vllm-to-sglang)

**中文对照**：要在 SGLang 中支持新模型，您只需在 [SGLang 模型目录](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models) 下添加一个文件。您可以参考现有的模型实现为您的模型创建一个新文件。对于大多数模型，您应该能够找到一个相似的模型作为起点（例如，从 Llama 开始）。另请参阅如何[从 vLLM 移植模型到 SGLang](#port-a-model-from-vllm-to-sglang)。

## How to Support a New Multimodal Large Language Model

To support a new multimodal large language model (MLLM) in SGLang, there are several key components in addition to the
standard LLM support:

1. **Register your new model as multimodal**:
   Extend `is_multimodal_model`
   in [model_config.py](https://github.com/sgl-project/sglang/blob/0ab3f437aba729b348a683ab32b35b214456efc7/python/sglang/srt/configs/model_config.py#L561)
   to return `True` for your model.

**中文对照**：1. **将您的新模型注册为多模态**：在 [model_config.py](https://github.com/sgl-project/sglang/blob/0ab3f437aba729b348a683ab32b35b214456efc7/python/sglang/srt/configs/model_config.py#L561) 中扩展 `is_multimodal_model` 函数，使其为您的模型返回 `True`。

2. **Register a new chat-template**:
   Only when your default chat-template is unable to accept images as input: Register a new chat template in [conversation.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/conversation.py) and the corresponding matching function.

**中文对照**：2. **注册新的对话模板**：仅当您的默认对话模板无法接受图像输入时：在 [conversation.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/conversation.py) 中注册新的对话模板及相应的匹配函数。

3. **Multimodal Data Processor**:
   Define a new `Processor` class that inherits from `BaseMultimodalProcessor` and register this processor as your
   model’s dedicated processor.
   See [multimodal_processor.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/multimodal/processors)
   for more details.

**中文对照**：3. **多模态数据处理器**：定义一个继承自 `BaseMultimodalProcessor` 的新 `Processor` 类，并将此处理器注册为您模型的专用处理器。详情请参阅 [multimodal_processor.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/multimodal/processors)。

4. **Handle Multimodal Tokens**:
   Implement a `pad_input_ids` function for your new model. In this function, multimodal tokens in the prompt should be
   expanded (if necessary) and padded with multimodal-data-hashes so that SGLang can recognize different multimodal data
   with `RadixAttention`.

**中文对照**：4. **处理多模态 Token**：为您的新模型实现 `pad_input_ids` 函数。在此函数中，提示中的多模态 token 应该被扩展（如有必要）并填充多模态数据哈希，以便 SGLang 能够通过 `RadixAttention` 识别不同的多模态数据。

5. **Handle Image Feature Extraction**:
   Implement a `get_image_feature` function for your new model, which extracts image features from raw image data and converts them into the embeddings used by the language model.

**中文对照**：5. **处理图像特征提取**：为您的新模型实现 `get_image_feature` 函数，该函数从原始图像数据中提取图像特征，并将其转换为语言模型使用的嵌入向量。

6. **Adapt to Vision Attention**:
   Adapt the multi-headed `Attention` of ViT with SGLang's `VisionAttention`.

**中文对照**：6. **适配视觉注意力**：使用 SGLang 的 `VisionAttention` 适配 ViT 的多头 `Attention`。

You can refer to [Qwen2VL](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_vl.py) or
other mllm implementations. These models demonstrate how to correctly handle both multimodal and textual inputs.

**中文对照**：您可以参考 [Qwen2VL](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_vl.py) 或其他多模态大语言模型实现。这些模型展示了如何正确处理多模态和文本输入。

## Testing and Debugging

Please note all your testing and benchmarking results in PR description.

**中文对照**：测试和调试

请在 PR 描述中注明您所有的测试和基准测试结果。

### Interactive Debugging

For interactive debugging, compare the outputs of Hugging Face/Transformers and SGLang. The following two commands
should give the same text output and very similar prefill logits:

**中文对照**：交互式调试

要进行交互式调试，请比较 Hugging Face/Transformers 和 SGLang 的输出。以下两个命令应该产生相同的文本输出和非常相似的预填充 logit：

- Get the reference output:
  ```bash
  python3 scripts/playground/reference_hf.py --model-path [new model] --model-type {text,mllm}
  ```
- Get the SGLang output:
  ```bash
  python3 -m sglang.bench_one_batch --correct --model [new model]
  ```

### Add the Model to the Test Suite

To ensure the new model is well maintained, add it to the test suite by including it in the `ALL_OTHER_MODELS` list in
the [test_generation_models.py](https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py)
file, test the new model on your local machine and report the results on demonstrative benchmarks (GSM8K, MMLU, MMMU,
MMMU-Pro, etc.) in your PR. \\
For VLMs, also include a test in `test_vision_openai_server_{x}.py` (e.g. [test_vision_openai_server_a.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_a.py), [test_vision_openai_server_b.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_b.py)).

**中文对照**：将模型添加到测试套件

为确保新模型得到良好维护，请将其添加到测试套件，方法是在 [test_generation_models.py](https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py) 文件的 `ALL_OTHER_MODELS` 列表中包含它，在本地机器上测试新模型，并在 PR 中报告演示基准测试（GSM8K、MMLU、MMMU、MMMU-Pro 等）的结果。对于视觉语言模型，还应在 `test_vision_openai_server_{x}.py` 中包含测试（例如 [test_vision_openai_server_a.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_a.py)、[test_vision_openai_server_b.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_b.py)）。

This is an example command to run to test a new model on your local machine:

**中文对照**：以下是在本地机器上测试新模型的示例命令：

```bash
ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others
```

### Benchmark

**中文对照**：基准测试

- **(Required) MMMU**: follow MMMU benchmark [README.md](https://github.com/sgl-project/sglang/blob/main/benchmark/mmmu/README.md) to get SGLang vs. HF Transformer accuracy comparison. The accuracy score from SGLang run should not be much lower than that from HF Transformer run. Similarly, follow https://docs.sglang.io/developer_guide/benchmark_and_profiling.html to get performance comparison: TTFT and throughput must meet or exceed baselines (e.g., HF Transformer).
- **(Optional) Other evals**: If you ran other evals, please note the results in PR description.

**中文对照**：- **（必需）MMMU**：按照 MMMU 基准测试 [README.md](https://github.com/sgl-project/sglang/blob/main/benchmark/mmmu/README.md) 获取 SGLang 与 HF Transformer 的准确率比较。SGLang 运行的准确率得分不应比 HF Transformer 运行低很多。同样，按照 https://docs.sglang.io/developer_guide/benchmark_and_profiling.html 进行性能比较：TTFT 和吞吐量必须达到或超过基线（例如 HF Transformer）。
- **（可选）其他评估**：如果您运行了其他评估，请在 PR 描述中注明结果。

## Port a Model from vLLM to SGLang

The [vLLM Models Directory](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) is a valuable
resource, as vLLM covers many models. SGLang reuses vLLM's interface and some layers, making it easier to port models
from vLLM to SGLang.

**中文对照**：从 vLLM 移植模型到 SGLang

[vLLM 模型目录](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) 是一个宝贵的资源，因为 vLLM 涵盖了许多模型。SGLang 重用了 vLLM 的接口和一些层，使从 vLLM 移植模型到 SGLang 变得更加容易。

To port a model from vLLM to SGLang:

**中文对照**：要从 vLLM 移植模型到 SGLang：

- Compare these two files for guidance:
  - [SGLang Llama Implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py)
  - [vLLM Llama Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)
- The major differences include:
  - **Replace vLLM's `Attention` with `RadixAttention`** (ensure you pass `layer_id` to `RadixAttention`).
  - **Replace vLLM's `LogitsProcessor` with SGLang's `LogitsProcessor`.**
  - **Replace the multi-headed `Attention` of ViT with SGLang's `VisionAttention`.**
  - **Replace other vLLM layers** (such as `RMSNorm`, `SiluAndMul`) with SGLang layers.
  - **Remove `Sample`.**
  - **Change the `forward()` functions** and add a `forward_batch()` method.
  - **Add `EntryClass`** at the end.
  - **Ensure that the new implementation uses only SGLang components** and does not rely on any vLLM components.

**中文对照**：- 比较以下两个文件以获取指导：
  - [SGLang Llama 实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py)
  - [vLLM Llama 实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)
- 主要差异包括：
  - **将 vLLM 的 `Attention` 替换为 `RadixAttention`**（确保将 `layer_id` 传递给 `RadixAttention`）。
  - **将 vLLM 的 `LogitsProcessor` 替换为 SGLang 的 `LogitsProcessor`**。
  - **将 ViT 的多头 `Attention` 替换为 SGLang 的 `VisionAttention`**。
  - **将其他 vLLM 层**（如 `RMSNorm`、`SiluAndMul`）替换为 SGLang 层。
  - **删除 `Sample`**。
  - **更改 `forward()` 函数**并添加 `forward_batch()` 方法。
  - **在末尾添加 `EntryClass`**。
  - **确保新实现只使用 SGLang 组件**，不依赖任何 vLLM 组件。

Note: make sure you add your new model to the supported models list in the supported models documentation.

**中文对照**：注意：请确保将您的新模型添加到支持模型文档中的支持模型列表中。

## Registering an External Model Implementation

In addition to the methods above, you can register your new model with the `ModelRegistry` before launching the server.
This allows you to integrate your model without modifying the source code.

**中文对照**：注册外部模型实现

除了上述方法外，您还可以在启动服务器之前向 `ModelRegistry` 注册您的新模型。这使您能够集成模型而无需修改源代码。

For example:

**中文对照**：例如：

```python
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.entrypoints.http_server import launch_server

# For a single model, add it to the registry:
ModelRegistry.models[model_name] = model_class

# For multiple models, you can imitate the import_model_classes() function:
from functools import lru_cache

@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    # Populate model_arch_name_to_cls with your new model classes.
    ...
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())

# Launch the server with your server arguments:
launch_server(server_args)
```

## Example: Implementing and Serving a Llama Wrapper Model

Below is an introductory, step-by-step walkthrough on how to implement a new model end-to-end in SGLang and then run it via the [Offline Engine](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/offline_engine_api.ipynb).

**中文对照**：示例：实现和提供 Llama 包装模型

以下是关于如何在 SGLang 中端到端实现新模型，然后通过[离线引擎](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/offline_engine_api.ipynb)运行它的介绍性、逐步演练。

### Implementing Our Model

To keep things simple, this new model will be a simple wrapper around [Llama 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), and our goal will be just to bias the output logits for each `forward` call by taking the square root of each individual logit.

**中文对照**：实现我们的模型

为简单起见，这个新模型将是对 [Llama 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 的简单包装，我们的目标只是通过取每个单独 logit 的平方根来对每次 `forward` 调用的输出 logit 进行偏置。

Let's start by defining our model in a file called `llama_wrapper.py`.
The first step is to import the necessary libraries from SRT, which is SGLang's internal backend.

**中文对照**：我们首先在一个名为 `llama_wrapper.py` 的文件中定义我们的模型。第一步是从 SRT（SGLang 的内部后端）导入必要的库。

```python
# In the file `llama_wrapper.py`

import torch
from transformers import LlamaConfig
from typing import Optional
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

from sglang.srt.models.llama import LlamaForCausalLM
```

Next, we declare a new `class` for our model and have it inherit from `LlamaForCausalLM`, which allows our model to access `LlamaForCausalLM`'s predefined modules and layers, such as `LlamaAttention` and `LlamaMLP`.
Note that almost all model implementations take in `config` and `quant_config` as arguments for their `__init__` method; `config` and `quant_config` are passed in via [`model_loader/loader.py`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_loader/loader.py#L219).
Because we have inherited from `LlamaForCausalLM`, we can pass our parameters directly to its constructor, which will set the member variables for us.

**中文对照**：接下来，我们为模型声明一个新的 `class`，并使其继承自 `LlamaForCausalLM`，这允许我们的模型访问 `LlamaForCausalLM` 的预定义模块和层，如 `LlamaAttention` 和 `LlamaMLP`。
请注意，几乎所有模型实现都将 `config` 和 `quant_config` 作为 `__init__` 方法的参数；`config` 和 `quant_config` 通过 [`model_loader/loader.py`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_loader/loader.py#L219) 传入。
因为我们继承了 `LlamaForCausalLM`，我们可以直接将参数传递给它的构造函数，它将为我们设置成员变量。

```python
class LlamaWrapper(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
```

Now, we want to define the `forward` method, which is what will be called at inference time.
Note that the signature for `forward` is essentially the same for any model; you can take a look at the other models defined in the [`models` directory](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/) for references.
To see where exactly `forward` is called in the SGLang runtime's internals, take a look at [`forward_decode`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1705) and [`forward_extend`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1724) in the [`ModelRunner` class](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py).

**中文对照**：现在，我们要定义 `forward` 方法，这就是在推理时会被调用的方法。
请注意，`forward` 的签名对任何模型来说基本相同；您可以查看 [`models` 目录](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/) 中定义的其他模型作为参考。
要查看 `forward` 在 SGLang 运行时的内部确切调用位置，请查看 [`ModelRunner` 类](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py) 中的 [`forward_decode`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1705) 和 [`forward_extend`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1724)。

```python
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
```

We now call the `__call__` method for `self.model` (which is a member variable that `LlamaForCausalLM` defines in its `__init__` method), which eventually calls `LlamaForCausalLM`'s `forward` method.
After that, we feed the `hidden_states` into our model's `LogitsProcessor` (again defined in `LlamaForCausalLM`).

**中文对照**：现在，我们调用 `self.model` 的 `__call__` 方法（这是 `LlamaForCausalLM` 在其 `__init__` 方法中定义的成员变量），它最终会调用 `LlamaForCausalLM` 的 `forward` 方法。
之后，我们将 `hidden_states` 输入到我们模型的 `LogitsProcessor`（同样在 `LlamaForCausalLM` 中定义）。

```python
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        res: LogitsProcessorOutput = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )
```

After receiving the logits for the next token, we can finally perform our biasing step.

**中文对照**：在收到下一个 token 的 logits 后，我们终于可以执行偏置步骤了。

```python
        orig_logits = res.next_token_logits
        res.next_token_logits = torch.where(
            orig_logits > 0,
            orig_logits.sqrt(),
            orig_logits
        )

        return res
```

Now, our `LlamaWrapper` model is created and ready to be served!

**中文对照**：现在，我们的 `LlamaWrapper` 模型已创建完成并准备好提供服务了！

### Serving Our Model Via SGLang's Offline Engine

The next step of this walkthrough involves hosting our new model offline, so that it can be served locally and without an HTTP server.

**中文对照**：通过 SGLang 的离线引擎提供我们的模型

本教程的下一步涉及离线托管我们的新模型，以便它可以在本地提供服务，而无需 HTTP 服务器。

First, create a new file called `run.py`.
Now, we must ensure that SGLang's `ModelRegistry` can find our model.
To do this, we first download the model's configuration and weights from Huggingface.

**中文对照**：首先，创建一个名为 `run.py` 的新文件。
现在，我们必须确保 SGLang 的 `ModelRegistry` 能够找到我们的模型。
为此，我们首先从 Huggingface 下载模型的配置和权重。

```python
# In the file `run.py`

import asyncio
from functools import lru_cache
from huggingface_hub import snapshot_download
from llama_wrapper import LlamaWrapper # Make sure to import our new model!
import sglang as sgl
from sglang.srt.models.registry import ModelRegistry

# Make sure to request access to this model on Huggingface, then export your
# `HF_TOKEN` to download the model snapshot
llama_dir = snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="./llama_ckpt",
)
```

Now that we have our model on disk, we want to point it to `LlamaWrapper` by changing the `architectures` field in `./llama_ckpt/config.json` to be `LlamaWrapper`.
That way, when we pass in the path of our model checkpoint to SGLang, it will know that we want to use "LlamaWrapper" instead of "LlamaForCausalLM" as our model.

**中文对照**：现在我们的模型已在磁盘上，我们希望通过将 `./llama_ckpt/config.json` 中的 `architectures` 字段更改为 `LlamaWrapper` 来指向它。
这样，当我们向 SGLang 传递模型检查点的路径时，它将知道我们要使用 "LlamaWrapper" 而不是 "LlamaForCausalLM" 作为我们的模型。

```python
{
  "architectures": [
   #  "LlamaForCausalLM"
    "LlamaWrapper"
  ],
  ...
}
```

However, if we don't link our `LlamaWrapper` class to the "LlamaWrapper" registry keyword, then SGLang won't be able to find our model.
Thus, to register our `LlamaWrapper`, we want to follow the steps in the above section titled "Registering an External Model Implementation".

**中文对照**：但是，如果我们不将 `LlamaWrapper` 类链接到 "LlamaWrapper" 注册表关键字，SGLang 将无法找到我们的模型。
因此，要注册我们的 `LlamaWrapper`，我们需要按照上面标题为"注册外部模型实现"部分中的步骤操作。

```python
@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {"LlamaWrapper": LlamaWrapper}
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())
```

Lastly, when we create our `Engine`, we just pass in the path to the local model directory.
Then, our `LlamaWrapper` is ready to be served; for this walkthrough, we will use SGLang `Engine`'s non-streaming asynchronous generation endpoint.

**中文对照**：最后，当我们创建 `Engine` 时，我们只需传入本地模型目录的路径。
然后，我们的 `LlamaWrapper` 就准备好了；对于本教程，我们将使用 SGLang `Engine` 的非流式异步生成端点。

```python
def main():
    llm = sgl.Engine(model_path="./llama_ckpt")
    sampling_params = {"temperature": 0.2, "top_k": 5}
    prompts = [
        "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
        "Provide a concise factual statement about France's capital city. The capital of France is",
        "Explain possible future trends in artificial intelligence. The future of AI is",
    ]

    asyncio.run(run_llm(llm, sampling_params, prompts))

    llm.shutdown()

async def run_llm(
    llm,
    sampling_params,
    prompts,
) -> None:
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")

if __name__ == "__main__":
    main()
```

Now, when we call `python run.py`, we will get the outputs of our newly created model!

**中文对照**：现在，当我们调用 `python run.py` 时，我们将获得新创建的模型的输出！

## Documentation

Add to table of supported models in [generative_models.md](../text_generation/generative_models.md) or [multimodal_language_models.md](../text_generation/multimodal_language_models.md)

**中文对照**：文档

添加到 [generative_models.md](../text_generation/generative_models.md) 或 [multimodal_language_models.md](../text_generation/multimodal_language_models.md) 中的支持模型表格中。

---

By following these guidelines, you can add support for new language models and multimodal large language models in
SGLang and ensure they are thoroughly tested and easily integrated into the system.

**中文对照**：通过遵循这些指南，您可以添加对新语言模型和多模态大语言模型在 SGLang 中的支持，并确保它们经过彻底测试并易于集成到系统中。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/` | 模型实现目录：每个架构一个文件（如 `llama.py`、`qwen2_vl.py`） |
| `python/sglang/srt/models/registry.py` | `ModelRegistry`：将架构名称（来自 `config.json`）映射到模型类 |
| `python/sglang/srt/configs/model_config.py` | `ModelConfig`：自动检测模型架构；`is_multimodal_model()` 用于多模态模型注册 |
| `python/sglang/srt/model_loader/loader.py` | 权重加载：将 `config` 和 `quant_config` 传递给模型的 `__init__()` |
| `python/sglang/srt/model_executor/model_runner.py` | `ModelRunner`：通过 `forward_decode()` 和 `forward_extend()` 调用模型的 `forward()` |
| `python/sglang/srt/layers/logits_processor.py` | `LogitsProcessor`：将隐藏状态转换为下一个 token 的 logits（替代 vLLM 的 `Sample`） |
| `python/sglang/srt/multimodal/processors/` | 多模态处理器：`BaseMultimodalProcessor` 子类，用于图像/视频特征提取 |
| `python/sglang/srt/conversation.py` | 对话模板注册表：为自定义多模态模型注册新的 chat template |

### 集成要点

- **模型注册**：在模型文件末尾添加 `EntryClass = YourModel`；`config.json` 中的架构名必须与注册表中的 key 匹配
- **从 vLLM 迁移的关键替换**：`Attention` → `RadixAttention`（需传入 `layer_id`），删除 `Sample`，替换 `LogitsProcessor`，ViT 的 `Attention` → `VisionAttention`
- **外部注册**：在调用 `launch_server()` 前执行 `ModelRegistry.models["MyModel"] = MyClass`，无需修改源码
- **测试**：`ONLY_RUN=ModelName python3 -m unittest test_generation_models.TestGenerationModels.test_others`
