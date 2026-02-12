# MiniMax M2.1/M2 Usage

[MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) and [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) are advanced large language models created by [MiniMax](https://www.minimax.io/).

**中文对照**：[MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) 和 [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) 是由 [MiniMax](https://www.minimax.io/) 创建的先进大型语言模型。

MiniMax-M2 series redefines efficiency for agents. It's a compact, fast, and cost-effective MoE model (230 billion total parameters with 10 billion active parameters) built for elite performance in coding and agentic tasks, all while maintaining powerful general intelligence. With just 10 billion activated parameters, MiniMax-M2 provides the sophisticated, end-to-end tool use performance expected from today's leading models, but in a streamlined form factor that makes deployment and scaling easier than ever.

**中文对照**：MiniMax-M2 系列重新定义了代理的效率。它是一个紧凑、快速且具有成本效益的 MoE 模型（总共 2300 亿参数，10 亿活跃参数），专为卓越的编码和代理任务性能而构建，同时保持强大的通用智能。仅需 10 亿激活参数，MiniMax-M2 就能提供当今领先模型所期望的复杂端到端工具使用性能，但采用了精简的外形，使部署和扩展比以往更加容易。

## Supported Models

**中文对照**：支持的模型

This guide applies to the following models. You only need to update the model name during deployment. The following examples use **MiniMax-M2**:

**中文对照**：本指南适用于以下模型。你只需在部署时更新模型名称。以下示例使用 **MiniMax-M2**：

- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
- [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)

## System Requirements

**中文对照**：系统要求

The following are recommended configurations; actual requirements should be adjusted based on your use case:

**中文对照**：以下是推荐配置；实际要求应根据你的使用情况进行调整：

- 4x 96GB GPUs: Supported context length of up to 400K tokens.

**中文对照**：- 4 个 96GB GPU：支持最高 400K 个令牌的上下文长度。

- 8x 144GB GPUs: Supported context length of up to 3M tokens.

**中文对照**：- 8 个 144GB GPU：支持最高 3M 个令牌的上下文长度。

## Deployment with Python

**中文对照**：使用 Python 部署

4-GPU deployment command:

**中文对照**：4 GPU 部署命令：

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2 \
    --tp-size 4 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

8-GPU deployment command:

**中文对照**：8 GPU 部署命令：

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2 \
    --tp-size 8 \
    --ep-size 8 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

## Testing Deployment

**中文对照**：测试部署

After startup, you can test the SGLang OpenAI-compatible API with the following command:

**中文对照**：启动后，你可以使用以下命令测试 SGLang OpenAI 兼容 API：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniMaxAI/MiniMax-M2",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Who won the world series in 2020?"}]}
        ]
    }'
```

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/minimax_m2.py` | MiniMax M2/M2.1 MoE 模型架构（230B 总参数，10B 激活参数） |
| `python/sglang/srt/function_call/minimax_m2_detector.py` | MiniMax M2 工具调用格式检测和解析 |

### 集成要点

- **服务器参数**：`--tp-size 4|8`、`--ep-size 8`（8-GPU 配置）、`--tool-call-parser minimax-m2`、`--reasoning-parser minimax-append-think`、`--trust-remote-code`
- **API**：兼容 OpenAI 的 `/v1/chat/completions` 端点
