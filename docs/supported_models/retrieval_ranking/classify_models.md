# Classification API

This document describes the `/v1/classify` API endpoint implementation in SGLang, which is compatible with vLLM's classification API format.

**中文对照**：分类 API

本文档描述了 SGLang 中 `/v1/classify` API 端点的实现，该实现与 vLLM 的分类 API 格式兼容。

## Overview

The classification API allows you to classify text inputs using classification models. This implementation follows the same format as vLLM's 0.7.0 classification API.

**中文对照**：概述

分类 API 允许您使用分类模型对文本输入进行分类。此实现遵循与 vLLM 0.7.0 分类 API 相同的格式。

## API Endpoint

```
POST /v1/classify
```

## Request Format

```json
{
  "model": "model_name",
  "input": "text to classify"
}
```

### Parameters

- `model` (string, required): The name of the classification model to use
- `input` (string, required): The text to classify
- `user` (string, optional): User identifier for tracking
- `rid` (string, optional): Request ID for tracking
- `priority` (integer, optional): Request priority

**中文对照**：### 参数

- `model`（字符串，必需）：要使用的分类模型的名称
- `input`（字符串，必需）：要分类的文本
- `user`（字符串，可选）：用于跟踪的用户标识符
- `rid`（字符串，可选）：用于跟踪的请求 ID
- `priority`（整数，可选）：请求优先级

## Response Format

```json
{
  "id": "classify-9bf17f2847b046c7b2d5495f4b4f9682",
  "object": "list",
  "created": 1745383213,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "index": 0,
      "label": "Default",
      "probs": [0.565970778465271, 0.4340292513370514],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

### Response Fields

- `id`: Unique identifier for the classification request
- `object`: Always "list"
- `created`: Unix timestamp when the request was created
- `model`: The model used for classification
- `data`: Array of classification results
  - `index`: Index of the result
  - `label`: Predicted class label
  - `probs`: Array of probabilities for each class
  - `num_classes`: Total number of classes
- `usage`: Token usage information
  - `prompt_tokens`: Number of input tokens
  - `total_tokens`: Total number of tokens
  - `completion_tokens`: Number of completion tokens (always 0 for classification)
  - `prompt_tokens_details`: Additional token details (optional)

**中文对照**：### 响应字段

- `id`：分类请求的唯一标识符
- `object`：始终为 "list"
- `created`：创建请求时的 Unix 时间戳
- `model`：用于分类的模型
- `data`：分类结果数组
  - `index`：结果的索引
  - `label`：预测的类别标签
  - `probs`：每个类别的概率数组
  - `num_classes`：类别总数
- `usage`：Token 使用信息
  - `prompt_tokens`：输入 token 的数量
  - `total_tokens`：token 总数
  - `completion_tokens`：完成 token 的数量（分类始终为 0）
  - `prompt_tokens_details`：额外的 token 详细信息（可选）

## Example Usage

### Using curl

```bash
curl -v "http://127.0.0.1:8000/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "Loved the new café—coffee was great."
  }'
```

### Using Python

```python
import requests
import json

# Make classification request
response = requests.post(
    "http://127.0.0.1:8000/v1/classify",
    headers={"Content-Type": "application/json"},
    json={
        "model": "jason9693/Qwen2.5-1.5B-apeach",
        "input": "Loved the new café—coffee was great."
    }
)

# Parse response
result = response.json()
print(json.dumps(result, indent=2))
```

## Supported Models

The classification API works with any classification model supported by SGLang, including:

### Classification Models (Multi-class)
- `LlamaForSequenceClassification` - Multi-class classification
- `Qwen2ForSequenceClassification` - Multi-class classification
- `Qwen3ForSequenceClassification` - Multi-class classification
- `BertForSequenceClassification` - Multi-class classification
- `Gemma2ForSequenceClassification` - Multi-class classification

**Label Mapping**: The API automatically uses the `id2label` mapping from the model's `config.json` file to provide meaningful label names instead of generic class names. If `id2label` is not available, it falls back to `LABEL_0`, `LABEL_1`, etc., or `Class_0`, `Class_1` as a last resort.

**中文对照**：### 支持的模型

分类 API 适用于 SGLang 支持的任何分类模型，包括：

### 分类模型（多类）
- `LlamaForSequenceClassification` - 多类分类
- `Qwen2ForSequenceClassification` - 多类分类
- `Qwen3ForSequenceClassification` - 多类分类
- `BertForSequenceClassification` - 多类分类
- `Gemma2ForSequenceClassification` - 多类分类

**标签映射**：API 自动使用模型 `config.json` 文件中的 `id2label` 映射来提供有意义的标签名称，而不是通用的类名。如果 `id2label` 不可用，它会回退到 `LABEL_0`、`LABEL_1` 等，或作为最后手段使用 `Class_0`、`Class_1`。

### Reward Models (Single score)
- `InternLM2ForRewardModel` - Single reward score
- `Qwen2ForRewardModel` - Single reward score
- `LlamaForSequenceClassificationWithNormal_Weights` - Special reward model

**Note**: The `/classify` endpoint in SGLang was originally designed for reward models but now supports all non-generative models. Our `/v1/classify` endpoint provides a standardized vLLM-compatible interface for classification tasks.

**中文对照**：### 奖励模型（单分数）
- `InternLM2ForRewardModel` - 单个奖励分数
- `Qwen2ForRewardModel` - 单个奖励分数
- `LlamaForSequenceClassificationWithNormal_Weights` - 特殊奖励模型

**注意**：SGLang 中的 `/classify` 端点最初是为奖励模型设计的，但现在支持所有非生成模型。我们的 `/v1/classify` 端点为分类任务提供了标准化的 vLLM 兼容接口。

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid request format or missing required fields
- `500 Internal Server Error`: Server-side processing error

Error response format:
```json
{
  "error": "Error message",
  "type": "error_type",
  "code": 400
}
```

**中文对照**：错误处理

API 返回适当的 HTTP 状态码和错误消息：

- `400 Bad Request`：无效的请求格式或缺少必需字段
- `500 Internal Server Error`：服务器端处理错误

错误响应格式：
```json
{
  "error": "Error message",
  "type": "error_type",
  "code": 400
}
```

## Implementation Details

The classification API is implemented using:

1. **Rust Model Gateway**: Handles routing and request/response models in `sgl-model-gateway/src/protocols/spec.rs`
2. **Python HTTP Server**: Implements the actual endpoint in `python/sglang/srt/entrypoints/http_server.py`
3. **Classification Service**: Handles the classification logic in `python/sglang/srt/entrypoints/openai/serving_classify.py`

**中文对照**：实现细节

分类 API 是使用以下组件实现的：

1. **Rust 模型网关**：在 `sgl-model-gateway/src/protocols/spec.rs` 中处理路由和请求/响应模型
2. **Python HTTP 服务器**：在 `python/sglang/srt/entrypoints/http_server.py` 中实现实际端点
3. **分类服务**：在 `python/sglang/srt/entrypoints/openai/serving_classify.py` 中处理分类逻辑

## Testing

Use the provided test script to verify the implementation:

```bash
python test_classify_api.py
```

**中文对照**：测试

使用提供的测试脚本来验证实现：

```bash
python test_classify_api.py
```

## Compatibility

This implementation is compatible with vLLM's classification API format, allowing seamless migration from vLLM to SGLang for classification tasks.

**中文对照**：兼容性

此实现与 vLLM 的分类 API 格式兼容，允许从 vLLM 无缝迁移到 SGLang 进行分类任务。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/entrypoints/http_server.py` | `/v1/classify` API 端点实现 |
| `python/sglang/srt/entrypoints/openai/serving_classify.py` | 分类服务逻辑：处理分类请求、计算类别概率 |
| `sgl-model-gateway/src/protocols/spec.rs` | Rust 模型网关：路由和请求/响应模型定义 |

### 集成要点

- **分类模型**：支持 `LlamaForSequenceClassification`、`Qwen2ForSequenceClassification`、`BertForSequenceClassification` 等多类别分类
- **奖励模型**：同时支持 `InternLM2ForRewardModel`、`Qwen2ForRewardModel` 等单分数奖励模型
- **标签映射**：自动使用模型 `config.json` 中的 `id2label` 映射提供有意义的标签名
- **兼容性**：与 vLLM 的分类 API 格式兼容，支持从 vLLM 无缝迁移
