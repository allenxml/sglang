# GPT OSS Usage

Please refer to [https://github.com/sgl-project/sglang/issues/8833](https://github.com/sgl-project/sglang/issues/8833).

## Responses API & Built-in Tools

### Responses API

GPT-OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built-in tool use. You can set reasoning level via `instructions`, e.g., "Reasoning: high" (also supports "medium" and "low") — levels: low (fast), medium (balanced), high (deep).

**中文对照**：GPT-OSS 与 OpenAI Responses API 兼容。使用带有 `model`、`instructions`、`input` 和可选 `tools` 的 `client.responses.create(...)` 来启用内置工具使用。你可以通过 `instructions` 设置推理级别，例如 "Reasoning: high"（还支持 "medium" 和 "low"）——级别：低（快速）、中等（平衡）、高（深度）。

### Built-in Tools

GPT-OSS can call built-in tools for web search and Python execution. You can use the demo tool server or connect to external MCP tool servers.

**中文对照**：GPT-OSS 可以调用内置工具进行网页搜索和 Python 执行。你可以使用演示工具服务器或连接外部 MCP 工具服务器。

#### Python Tool

- Executes short Python snippets for calculations, parsing, and quick scripts.
- By default runs in a Docker-based sandbox. To run on the host, set `PYTHON_EXECUTION_BACKEND=UV` (this executes model-generated code locally; use with care).
- Ensure Docker is available if you are not using the UV backend. It is recommended to run `docker pull python:3.11` in advance.

**中文对照**：- 执行简短的 Python 代码片段用于计算、解析和快速脚本。
- 默认在基于 Docker 的沙箱中运行。要在主机上运行，设置 `PYTHON_EXECUTION_BACKEND=UV`（这会在本地执行模型生成的代码；请谨慎使用）。
- 如果不使用 UV 后端，请确保 Docker 可用。建议提前运行 `docker pull python:3.11`。

#### Web Search Tool

- Uses the Exa backend for web search.
- Requires an Exa API key; set `EXA_API_KEY` in your environment. Create a key at `https://exa.ai`.

**中文对照**：- 使用 Exa 后端进行网页搜索。
- 需要 Exa API 密钥；在你的环境中设置 `EXA_API_KEY`。在 `https://exa.ai` 创建一个密钥。

### Tool & Reasoning Parser

- We support OpenAI Reasoning and Tool Call parser, as well as our SGLang native api for tool call and reasoning. Refer to [reasoning parser](../advanced_features/separate_reasoning.ipynb) and [tool call parser](../advanced_features/function_calling.ipynb) for more details.

**中文对照**：- 我们支持 OpenAI Reasoning 和 Tool Call 解析器，以及 SGLang 原生的工具调用和推理 API。更多详情请参阅[推理解析器](../advanced_features/separate_reasoning.ipynb)和[工具调用解析器](../advanced_features/function_calling.ipynb)。


## Notes

- Use **Python 3.12** for the demo tools. And install the required `gpt-oss` packages.
- The default demo integrates the web search tool (Exa backend) and a demo Python interpreter via Docker.
- For search, set `EXA_API_KEY`. For Python execution, either have Docker available or set `PYTHON_EXECUTION_BACKEND=UV`.

**中文对照**：- 对演示工具使用 **Python 3.12**。并安装所需的 `gpt-oss` 包。
- 默认演示集成了网页搜索工具（Exa 后端）和通过 Docker 的演示 Python 解释器。
- 对于搜索，设置 `EXA_API_KEY`。对于 Python 执行，要么确保 Docker 可用，要么设置 `PYTHON_EXECUTION_BACKEND=UV`。

Examples:
```bash
export EXA_API_KEY=YOUR_EXA_KEY
# Optional: run Python tool locally instead of Docker (use with care)
export PYTHON_EXECUTION_BACKEND=UV
```

Launch the server with the demo tool server:

**中文对照**：使用演示工具服务器启动服务器：

```bash
python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --tool-server demo \
  --tp 2
```

For production usage, sglang can act as an MCP client for multiple services. An [example tool server](https://github.com/openai/gpt-oss/tree/main/gpt-oss-mcp-server) is provided. Start the servers and point sglang to them:

**中文对照**：对于生产使用，sglang 可以作为多个服务的 MCP 客户端。提供了一个[示例工具服务器](https://github.com/openai/gpt-oss/tree/main/gpt-oss-mcp-server)。启动服务器并将 sglang 指向它们：
```bash
mcp run -t sse browser_server.py:mcp
mcp run -t sse python_server.py:mcp

python -m sglang.launch_server ... --tool-server ip-1:port-1,ip-2:port-2
```
The URLs should be MCP SSE servers that expose server information and well-documented tools. These tools are added to the system prompt so the model can use them.

**中文对照**：URL 应该是公开服务器信息且工具文档完善的 MCP SSE 服务器。这些工具被添加到系统提示中，以便模型可以使用它们。

## Speculative Decoding

SGLang supports speculative decoding for GPT-OSS models using EAGLE3 algorithm. This can significantly improve decoding speed, especially for small batch sizes.

**中文对照**：SGLang 支持使用 EAGLE3 算法对 GPT-OSS 模型进行推测解码。这可以显著提高解码速度，特别是对于小批次大小。

**Usage**:

**中文对照**：**用法**：

Add `--speculative-algorithm EAGLE3` along with the draft model path.

**中文对照**：添加 `--speculative-algorithm EAGLE3` 以及草稿模型路径。
```bash
python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 \
  --tp 2
```

### Quick Demo

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-123456"
)

tools = [
    {"type": "code_interpreter"},
    {"type": "web_search_preview"},
]

# Reasoning level example
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant."
    reasoning_effort="high" # Supports high, medium, or low
    input="In one sentence, explain the transformer architecture.",
)
print("====== reasoning: high ======")
print(response.output_text)

# Test python tool
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant, you could use python tool to execute code.",
    input="Use python tool to calculate the sum of 29138749187 and 29138749187", # 58,277,498,374
    tools=tools
)
print("====== test python tool ======")
print(response.output_text)

# Test browser tool
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant, you could use browser to search the web",
    input="Search the web for the latest news about Nvidia stock price",
    tools=tools
)
print("====== test browser tool ======")
print(response.output_text)
```

Example output:
```
====== test python tool ======
The sum of 29,138,749,187 and 29,138,749,187 is **58,277,498,374**.
====== test browser tool ======
**Recent headlines on Nvidia (NVDA) stock**

| Date (2025) | Source | Key news points | Stock-price detail |
|-------------|--------|----------------|--------------------|
| **May 13** | Reuters | The market data page shows Nvidia trading "higher" at **$116.61** with no change from the previous close. | **$116.61** - latest trade (delayed 15 min) |
| **Aug 18** | CNBC | Morgan Stanley kept an **overweight** rating and lifted its price target to **$206** (up from $200), implying a 14% upside from the Friday close. The firm notes Nvidia shares have already **jumped 34% this year**. | No exact price quoted, but the article signals strong upside expectations |
| **Aug 20** | The Motley Fool | Nvidia is set to release its Q2 earnings on Aug 27. The article lists the **current price of $175.36**, down 0.16% on the day (as of 3:58 p.m. ET). | **$175.36** - current price on Aug 20 |

**What the news tells you**

* Nvidia's share price has risen sharply this year - up roughly a third according to Morgan Stanley - and analysts are still raising targets (now $206).
* The most recent market quote (Reuters, May 13) was **$116.61**, but the stock has surged since then, reaching **$175.36** by mid-August.
* Upcoming earnings on **Aug 27** are a focal point; both the Motley Fool and Morgan Stanley expect the results could keep the rally going.

**Bottom line:** Nvidia's stock is on a strong upward trajectory in 2025, with price targets climbing toward $200-$210 and the market price already near $175 as of late August.

```

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/gpt_oss.py` | GPT-OSS 模型架构，支持内置工具使用 |
| `python/sglang/srt/entrypoints/http_server.py` | Responses API 端点（`/v1/responses`），兼容 OpenAI 格式 |
| `python/sglang/srt/speculative/eagle_worker.py` | GPT-OSS 的 EAGLE3 推测解码 |
| `python/sglang/srt/tool_server/` | MCP 工具服务器集成，用于网页搜索和 Python 执行 |

### 集成要点

- **服务器参数**：`--tool-server demo`（内置工具）或 `--tool-server ip:port`（外部 MCP 服务器）、`--speculative-algorithm EAGLE3 --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16`
- **环境变量**：`EXA_API_KEY`（网页搜索）、`PYTHON_EXECUTION_BACKEND=UV`（本地 Python 执行）
- **API**：通过 `client.responses.create()` 使用 OpenAI Responses API，支持 `reasoning_effort` 和 `tools` 参数
