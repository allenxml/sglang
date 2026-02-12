# 🚂 一次推理请求的完整旅程

> **目标**：跟随一个真实的用户请求，从发出到收到回答，经过的每一个"站点"。
> 这是本指南中最重要的一篇——理解了请求流程，就理解了 SGLang 的核心工作原理。

---

## 🎬 场景设定

假设你向 SGLang 服务发送了这样一个请求：

```json
POST /v1/chat/completions
{
  "model": "meta-llama/Llama-3-8B",
  "messages": [
    {"role": "user", "content": "什么是人工智能？"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

现在，让我们跟着这个请求，走完它的 10 个站点。

---

## 🗺️ 旅程总览

```
站点 1     站点 2      站点 3      站点 4       站点 5
HTTP ──▶ 参数解析 ──▶ 分词 ──▶ ZMQ发送 ──▶ 调度入队
接收      验证        编码        传递         排队等候
  │
  │  ┌───────────────────────────────────────────────┘
  │  │
  │  ▼
  │  站点 6      站点 7      站点 8       站点 9      站点 10
  │  组批 ──▶ GPU推理 ──▶ 采样 ──▶ 反分词 ──▶ 返回
  │  打包      前向计算     选词         解码         响应
  │                                                    │
  └────────────────────────────────────────────────────┘
```

---

## 🚏 站点 1：HTTP 接收

**位置**：`python/sglang/srt/entrypoints/http_server.py`
**比喻**：餐厅前台接到顾客的电话订单

```python
# FastAPI 路由接收 HTTP POST 请求
@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(request):
    ...
```

**发生了什么**：
1. FastAPI 框架接收到 HTTP POST 请求
2. 解析 JSON 请求体
3. 调用处理函数

**输入**：原始 HTTP 请求
**输出**：Python 字典（解析后的参数）

---

## 🚏 站点 2：参数解析与验证

**位置**：`python/sglang/srt/entrypoints/http_server.py` → `v1_chat_completions()`
**比喻**：前台核实订单——"您点的菜我们有吗？数量对吗？"

**发生了什么**：
1. 将 OpenAI 格式的参数转换为 SGLang 内部格式
2. 把 `messages`（对话历史）拼接成完整的 prompt 文本
3. 应用聊天模板（Chat Template）——每个模型有自己的格式
4. 创建 `GenerateReqInput` 对象

```
用户消息: [{"role": "user", "content": "什么是人工智能？"}]
    ↓ 应用聊天模板
完整 prompt: "<|begin_of_text|><|user|>\n什么是人工智能？<|assistant|>\n"
    ↓
GenerateReqInput(text="<|begin_of_text|>...", sampling_params={...})
```

---

## 🚏 站点 3：分词（Tokenization）

**位置**：`python/sglang/srt/managers/tokenizer_manager.py` → `generate_request()`
**比喻**：把中文菜名翻译成厨房用的编号——"宫保鸡丁"→ "#A23"

**发生了什么**：
1. TokenizerManager 接收到 `GenerateReqInput`
2. 使用分词器（Tokenizer）将文本转为 Token ID 序列
3. 为请求分配唯一的请求 ID（rid）
4. 创建一个"等待槽位"（asyncio.Event），等结果回来

```
"<|begin_of_text|><|user|>\n什么是人工智能？<|assistant|>\n"
    ↓ 分词器
[128000, 128006, 882, 128007, 10, 什, 么, 是, 人, 工, 智, 能, ？, 128006, 78191, 128007, 10]
```

> 💡 **注意**：特殊标记如 `<|begin_of_text|>` 也有对应的 Token ID（如 128000），它们帮助模型理解文本结构。

---

## 🚏 站点 4：ZMQ 发送

**位置**：`python/sglang/srt/managers/tokenizer_manager.py`
**比喻**：前台通过对讲机把订单传给后厨

**发生了什么**：
1. 将 Token 化后的请求序列化
2. 通过 ZMQ（进程间通信）发送给 Scheduler 进程
3. TokenizerManager 开始等待结果返回

```
主进程                          Scheduler 进程
TokenizerManager ──ZMQ──▶ Scheduler 接收队列
    │
    └── 等待结果...（异步等待）
```

---

## 🚏 站点 5：调度入队

**位置**：`python/sglang/srt/managers/scheduler.py` → `process_input_requests()`
**比喻**：餐厅经理把新订单放到待处理队列上

**发生了什么**：
1. Scheduler 从 ZMQ 通道接收到新请求
2. 创建 `Req` 对象（内部请求表示）
3. 将请求放入等待队列（`waiting_queue`）
4. 尝试 Radix Cache 前缀匹配——看看之前有没有处理过类似的前缀

```
waiting_queue（等待队列）:
┌──────┬──────┬──────┬──────────┐
│ Req1 │ Req2 │ Req3 │ 新请求 ← │  入队
└──────┴──────┴──────┴──────────┘
```

---

## 🚏 站点 6：组成批次（Batching）

**位置**：`python/sglang/srt/managers/scheduler.py` → `get_next_batch_to_run()`
**比喻**：经理决定"这一轮同时做哪几桌的菜"

**发生了什么**：
1. Scheduler 检查当前资源（GPU 显存、KV Cache 空间）
2. 从等待队列中选取请求，组成一个批次（Batch）
3. 区分 Prefill 请求（新来的）和 Decode 请求（正在生成中的）
4. 创建 `ScheduleBatch` 对象

**关键决策**：
- **Continuous Batching**（连续批处理）：不用等一批全做完再开始下一批
- 已完成 Prefill 的请求可以直接开始 Decode
- 新请求可以随时加入正在运行的批次

```
当前批次:
┌─────────────────────────────────────┐
│  Decode 请求: Req1, Req2, Req3      │  ← 正在逐词生成
│  Prefill 请求: 新请求               │  ← 刚进来，需要先读题
└─────────────────────────────────────┘
```

---

## 🚏 站点 7：GPU 推理（Forward Pass）

**位置**：`python/sglang/srt/model_executor/model_runner.py`
**比喻**：厨师拿到菜单，开始炒菜

**发生了什么**：
1. ModelRunner 接收到批次数据
2. 准备输入张量（Tensor）——Token ID、位置信息、注意力掩码等
3. 执行模型的前向传播：
   - **Prefill 阶段**：一次性处理所有输入 Token（"通读题目"）
   - **Decode 阶段**：只处理最新的一个 Token（"写下一个字"）
4. 输出：每个请求的下一个 Token 的概率分布（logits）

```
输入: Token ID 序列 + KV Cache
    ↓
┌──────────────────┐
│   模型前向传播     │  ← 在 GPU 上执行，大量矩阵乘法
│   (Transformer)   │
└────────┬─────────┘
         ↓
输出: logits（每个词的概率分布）
    例: {"的": 0.15, "是": 0.12, "人工": 0.08, ...}
```

> 💡 **KV Cache 的作用**：Decode 阶段不需要重新计算之前所有 Token，因为之前的中间结果已经存在 KV Cache 里了。这就像你做数学题时在草稿纸上记下的中间步骤——下一步可以直接用，不用从头算。

---

## 🚏 站点 8：采样（Sampling）

**位置**：`python/sglang/srt/layers/sampler.py`（由 ModelRunner 调用）
**比喻**：从菜单上选出这次要上的菜

**发生了什么**：
1. 拿到 logits（概率分布）
2. 应用 Temperature（温度）：调整概率的"平坦程度"
   - Temperature 高 → 更随机（"有创意"）
   - Temperature 低 → 更确定（"更保守"）
3. 应用 Top-P / Top-K：只从最可能的几个词中选
4. 按概率随机抽取一个 Token

```
logits → Temperature(0.7) → Top-P(0.9) → 采样 → Token ID: 42
                                                    ("人" 字)
```

---

## 🚏 站点 9：反分词 + 完成检查

**位置**：`python/sglang/srt/managers/detokenizer_manager.py`
**比喻**：把编号翻译回菜名，准备上菜

**发生了什么**：
1. 新生成的 Token ID 被发送到 DetokenizerManager
2. 检查是否生成了结束标记（EOS Token）或达到了 `max_tokens`
3. 如果未完成：回到站点 6，继续下一轮 Decode
4. 如果完成：将所有 Token ID 转换为文本

```
循环过程（站点 6→7→8→9→6→...）:

第 1 轮: → "人"
第 2 轮: → "工"
第 3 轮: → "智"
第 4 轮: → "能"
第 5 轮: → "是"
...
第 N 轮: → "<EOS>" ← 生成结束标记，停止！

完整输出: "人工智能是一种模拟人类智能的技术..."
```

> 💡 **流式输出**：如果用户选择了流式模式（stream=true），每生成几个 Token 就会先发回去一部分，用户可以看到回答像打字一样逐步出现——这就是 ChatGPT 的"打字效果"。

---

## 🚏 站点 10：返回结果

**位置**：`python/sglang/srt/entrypoints/http_server.py`
**比喻**：传菜员把做好的菜端到顾客桌上

**发生了什么**：
1. DetokenizerManager 将完成的文本结果发送给 TokenizerManager
2. TokenizerManager 之前等待的"槽位"被唤醒
3. 将结果包装成 OpenAI 兼容的响应格式
4. 通过 HTTP 返回给用户

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "人工智能（Artificial Intelligence，简称AI）是一种模拟人类智能的技术..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 17,
    "completion_tokens": 85,
    "total_tokens": 102
  }
}
```

---

## 📊 完整时间线

```
时间 ──────────────────────────────────────────────────────────▶

站点1  站点2  站点3  站点4  站点5  站点6  站点7  站点8  站点9  站点10
HTTP → 解析 → 分词 → ZMQ → 入队 → 组批 → GPU → 采样 → 反词 → 返回
│<── ~1ms ──>│<─ ~1ms ─>│      │<────── 重复 N 次 ──────>│
                                      （每次 ~10-50ms）
```

**总耗时**：
- 首个 Token（Time To First Token, TTFT）：~50-200ms
- 后续每个 Token：~10-50ms
- 100 个 Token 的完整回答：~1-5 秒

---

## 🔑 关键洞察

1. **大部分时间花在 GPU 上**：站点 7（GPU 推理）是最耗时的环节
2. **Decode 是循环的**：生成 N 个 Token 需要循环 N 次（站点 6→7→8→9）
3. **批处理是性能关键**：多个请求共享一次 GPU 计算，大幅提高吞吐量
4. **KV Cache 避免重复计算**：每轮只计算新的一个 Token，之前的结果已缓存

---

## 下一步

- 深入了解分词 → [分词与反分词](05-tokenization.md)
- 深入了解调度 → [调度系统](06-scheduling.md)
- 深入了解 GPU 推理 → [模型推理](07-model-inference.md)
- 返回目录 → [README](README.md)
