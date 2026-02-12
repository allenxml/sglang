## Launch GLM-4.5 / GLM-4.6 / GLM-4.7 with SGLang

To serve GLM-4.5 / GLM-4.6 FP8 models on 8xH100/H200 GPUs:

```bash
python3 -m sglang.launch_server --model zai-org/GLM-4.6-FP8 --tp 8
```

### EAGLE Speculative Decoding

**中文对照**：EAGLE 推测解码

**Description**: SGLang has supported GLM-4.5 / GLM-4.6 models
with [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding).

**中文对照**：**描述**：SGLang 通过 [EAGLE 推测解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding) 支持 GLM-4.5 / GLM-4.6 模型。

**Usage**:

**中文对照**：**用法**：
Add arguments `--speculative-algorithm`, `--speculative-num-steps`, `--speculative-eagle-topk` and
`--speculative-num-draft-tokens` to enable this feature. For example:

``` bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.6-FP8 \
  --tp-size 8 \
  --tool-call-parser glm45  \
  --reasoning-parser glm45  \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3  \
  --speculative-eagle-topk 1  \
  --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.9 \
  --served-model-name glm-4.6-fp8 \
  --enable-custom-logit-processor
```

```{tip}
To enable the experimental overlap scheduler for EAGLE speculative decoding, set the environment variable `SGLANG_ENABLE_SPEC_V2=1`. This can improve performance by enabling overlap scheduling between draft and verification stages.
```

### Thinking Budget for GLM-4.5 / GLM-4.6

**中文对照**：GLM-4.5 / GLM-4.6 的思考预算

**Note**: For GLM-4.7, `--tool-call-parser` should be set to `glm47`, for GLM-4.5 and GLM-4.6, it should be set to `glm45`.

**中文对照**：**注意**：对于 GLM-4.7，`--tool-call-parser` 应设置为 `glm47`，对于 GLM-4.5 和 GLM-4.6，应设置为 `glm45`。

In SGLang, we can implement thinking budget with `CustomLogitProcessor`.

**中文对照**：在 SGLang 中，我们可以使用 `CustomLogitProcessor` 实现思考预算。

Launch a server with `--enable-custom-logit-processor` flag on.

**中文对照**：使用 `--enable-custom-logit-processor` 标志启动服务器。

Sample Request:

**中文对照**：示例请求：

```python
import openai
from rich.pretty import pprint
from sglang.srt.sampling.custom_logit_processor import Glm4MoeThinkingBudgetLogitProcessor


client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="*")
response = client.chat.completions.create(
    model="zai-org/GLM-4.6",
    messages=[
        {
            "role": "user",
            "content": "Question: Is Paris the Capital of France?",
        }
    ],
    max_tokens=1024,
    extra_body={
        "custom_logit_processor": Glm4MoeThinkingBudgetLogitProcessor().to_str(),
        "custom_params": {
            "thinking_budget": 512,
        },
    },
)
pprint(response)
```

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/glm4.py` | GLM-4.5/4.6/4.7 模型架构 |
| `python/sglang/srt/function_call/glm45_detector.py` | GLM-4.5/4.6 工具调用格式检测和解析 |
| `python/sglang/srt/sampling/custom_logit_processor.py` | `Glm4MoeThinkingBudgetLogitProcessor` 用于思考预算控制 |
| `python/sglang/srt/speculative/eagle_worker.py` | GLM 模型的 EAGLE 推测解码 |

### 集成要点

- **服务器参数**：`--tool-call-parser glm45`（GLM-4.5/4.6）或 `glm47`（GLM-4.7）、`--reasoning-parser glm45`、`--enable-custom-logit-processor`
- **推测解码**：`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
- **思考预算**：通过 `Glm4MoeThinkingBudgetLogitProcessor` 配合 `custom_params.thinking_budget`
