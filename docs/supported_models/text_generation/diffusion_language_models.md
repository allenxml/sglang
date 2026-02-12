# Diffusion Language Models

> This page covers **text generation** using diffusion-based LLMs. For **image and video generation**, see [Diffusion Models](../image_generation/diffusion_models.md).

**中文对照**：扩散语言模型

> 本页介绍使用基于扩散的 LLM 进行**文本生成**。有关**图像和视频生成**，请参阅[扩散模型](../image_generation/diffusion_models.md)。

Diffusion language models have shown promise for non-autoregressive text generation with parallel decoding capabilities. Unlike auto-regressive language models, different diffusion language models require different decoding strategies.

**中文对照**：扩散语言模型在具有并行解码功能的非自回归文本生成方面已显示出前景。与自回归语言模型不同，不同的扩散语言模型需要不同的解码策略。

## Example Launch Command

SGLang supports different DLLM algorithms such as `LowConfidence` and `JointThreshold`.

```shell
python3 -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \ # example HF/local path
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config ./config.yaml \ # Optional. Uses the algorithm's default if not set.
  --host 0.0.0.0 \
  --port 30000
```

**中文对照**：示例启动命令

SGLang 支持不同的 DLLM 算法，如 `LowConfidence` 和 `JointThreshold`。

## Example Configuration File

Depending on the algorithm selected, the configuration parameters vary.

LowConfidence Config:

```yaml
# Confidence threshold for accepting predicted tokens
# - Higher values: More conservative, better quality but slower
# - Lower values: More aggressive, faster but potentially lower quality
# Range: 0.0 - 1.0
threshold: 0.95

# Default: 32, for LLaDA2MoeModelLM
block_size: 32
```

JointThreshold Config:

```yaml
# Decoding threshold for Mask-to-Token (M2T) phase
# - Higher values: More conservative, better quality but slower
# - Lower values: More aggressive, faster but potentially lower quality
# Range: 0.0 - 1.0
threshold: 0.5
# Decoding threshold for Token-to-Token (T2T) phase
# Range: 0.0 - 1.0
# Setting to 0.0 allows full editing (recommended for most cases).
edit_threshold: 0.0
# Max extra T2T steps after all masks are removed. Prevents infinite loops.
max_post_edit_steps: 16
# 2-gram repetition penalty (default 0).
# An empirical value of 3 is often sufficient to mitigate most repetitions.
penalty_lambda: 0
```

**中文对照**：示例配置文件

根据所选算法，配置参数会有所不同。

LowConfidence 配置：

**中文对照**：LowConfidence 配置

JointThreshold 配置：

**中文对照**：JointThreshold 配置

## Example Client Code Snippet

Just like other supported models, diffusion language models can be used via the REST API or Python client.

Python client example for making a generation request to the launched server:

```python
import sglang as sgl

def main():
    llm = sgl.Engine(model_path="inclusionAI/LLaDA2.0-mini",
                     dllm_algorithm="LowConfidence",
                     max_running_requests=1,
                     trust_remote_code=True)

    prompts = [
        "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write a brief introduction of the great wall <|role_end|><role>ASSISTANT</role>"
    ]

    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 1024,
    }

    outputs = llm.generate(prompts, sampling_params)
    print(outputs)

if __name__ == '__main__':
    main()
```

Curl example for making a generation request to the launched server:

```bash
curl -X POST "http://127.0.0.1:30000/generate" \
     -H "Content-Type: application/json" \
     -d '{
        "text": [
            "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write the number from 1 to 128 <|role_end|><role>ASSISTANT</role>",
            "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write a brief introduction of the great wall <|role_end|><role>ASSISTANT</role>"
        ],
        "stream": true,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1024
        }
    }'
```

**中文对照**：客户端代码示例

与其他支持的模型一样，扩散语言模型可以通过 REST API 或 Python 客户端使用。

Python 客户端示例，用于向启动的服务器发出生成请求：

**中文对照**：Python 客户端示例

Curl 示例，用于向启动的服务器发出生成请求：

**中文对照**：Curl 示例

## Supported Models

Below the supported models are summarized in a table.

| Model Family               | Example Model                | Description                                                                                          |
| -------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| **LLaDA2.0 (mini, flash)** | `inclusionAI/LLaDA2.0-flash` | LLaDA2.0-flash is a diffusion language model featuring a 100B Mixture-of-Experts (MoE) architecture. |

**中文对照**：支持的模型

下表总结了支持的模型。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/` | 扩散语言模型实现：支持 LLaDA2.0 等非自回归文本生成架构 |
| `python/sglang/srt/server_args.py` | `--dllm-algorithm` 和 `--dllm-algorithm-config` 命令行参数：选择解码算法及其配置 |

### 集成要点

- **解码算法**：支持 `LowConfidence`（基于置信度阈值）和 `JointThreshold`（联合阈值，区分 M2T 和 T2T 阶段）
- **配置文件**：通过 YAML 文件配置算法参数（如 `threshold`、`block_size`、`edit_threshold` 等）
- **使用方式**：与标准模型相同的 REST API 和 Python SDK 接口；通过 `sglang.Engine` 或 HTTP 服务使用
- **与自回归模型的区别**：扩散语言模型支持并行解码，不同模型需要不同的解码策略
