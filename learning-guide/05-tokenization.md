# 🔤 分词与反分词

> **目标**：理解 Token 的概念、分词器如何工作、TokenizerManager 和 DetokenizerManager 的角色。

---

## 1. 什么是 Token？

### 比喻：乐高积木 🧱

想象你有一盒乐高积木，每块积木有一个编号。你可以用这些编号来"拼"出任何东西：

```
"人工智能" → [积木#1234, 积木#5678]
"你好世界" → [积木#2345, 积木#6789, 积木#3456]
```

Token 就是大语言模型的"乐高积木"——文字被拆成一块块 Token，每个 Token 用一个数字编号（Token ID）表示。

### 为什么需要 Token？

计算机（尤其是 GPU）只能处理数字，不能直接处理文字。所以需要：
1. **分词**（Tokenization）：文字 → 数字序列
2. **反分词**（Detokenization）：数字序列 → 文字

### Token 的粒度

Token 不一定是一个完整的字或词：

```
英文示例:
"Hello world" → ["Hello", " world"]                    # 2 个 Token
"unhappiness" → ["un", "happiness"]                    # 2 个 Token
"ChatGPT"     → ["Chat", "G", "PT"]                    # 3 个 Token

中文示例:
"你好"   → ["你", "好"]                                # 2 个 Token
"人工智能" → ["人工", "智能"]  或  ["人", "工", "智能"]   # 取决于分词器
```

> 💡 **经验法则**：英文大约每 4 个字符 ≈ 1 个 Token；中文大约每 1-2 个字 ≈ 1 个 Token。

---

## 2. 分词器（Tokenizer）

### 分词器是什么？

分词器是一个"翻译词典"，它知道如何把文字映射到数字。

```
分词器的内部（简化版）：
{
    "你": 12345,
    "好": 67890,
    "人工": 11111,
    "智能": 22222,
    "<|begin_of_text|>": 128000,   ← 特殊标记
    "<|end_of_text|>": 128001,     ← 结束标记
    ...
}
```

每个模型有自己的分词器——就像不同国家的人用不同的"翻译词典"。

### 常见的分词算法

| 算法 | 特点 | 使用者 |
|------|------|-------|
| BPE（字节对编码） | 从字符出发，合并高频组合 | GPT 系列、LLaMA |
| SentencePiece | BPE 的改良版，支持多语言 | Qwen、Mistral |
| WordPiece | 类似 BPE，但选择策略不同 | BERT |

---

## 3. TokenizerManager 详解

**文件**：`python/sglang/srt/managers/tokenizer_manager.py`

### 职责

TokenizerManager 不只是"分词"那么简单，它是请求进入系统的"总入口"：

```
用户请求 → TokenizerManager
              │
              ├── 1. 验证请求参数
              ├── 2. 应用聊天模板（Chat Template）
              ├── 3. 执行分词（文字 → Token ID）
              ├── 4. 分配请求 ID
              ├── 5. 通过 ZMQ 发送给 Scheduler
              ├── 6. 等待结果返回
              └── 7. 将结果交给用户
```

### 聊天模板（Chat Template）

不同模型对输入格式有不同要求。聊天模板就是把"用户消息列表"转换为模型期望的"原始文本"格式：

```
输入（用户提供的）:
messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"}
]

输出（LLaMA 3 的格式）:
"<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是一个助手<|eot_id|><|start_header_id|>user<|end_header_id|>
你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"
```

> 💡 **比喻**：就像不同国家的信件格式不同——有的要先写收件人，有的要先写日期。聊天模板确保"信件格式"正确，模型才能正确理解。

---

## 4. DetokenizerManager 详解

**文件**：`python/sglang/srt/managers/detokenizer_manager.py`

### 职责

反分词器把 GPU 生成的 Token ID 转换回人类可读的文字：

```
Token ID 序列: [1234, 5678, 9012, ...]
      │
      ▼ DetokenizerManager
      │
文字输出: "人工智能是一种..."
```

### 流式输出（Streaming）

这是 DetokenizerManager 最重要的特性之一：

**非流式**（等全部生成完再返回）：
```
[等待 3 秒...]
"人工智能是一种模拟人类智能的技术，它包括机器学习、深度学习等多个子领域。"
```

**流式**（边生成边返回）：
```
[0.1秒] "人工"
[0.2秒] "智能"
[0.3秒] "是一种"
[0.4秒] "模拟"
...（像打字机一样逐步出现）
```

> 💡 **用户体验**：流式输出让用户感觉响应更快——虽然总时间相同，但用户不需要盯着空白屏幕等待。这就是 ChatGPT 回答时"一个字一个字蹦出来"的效果。

### 流式输出的工作原理

```
Scheduler ──生成 Token#1──▶ DetokenizerManager ──"人工"──▶ 用户
Scheduler ──生成 Token#2──▶ DetokenizerManager ──"智能"──▶ 用户
Scheduler ──生成 Token#3──▶ DetokenizerManager ──"是"──▶ 用户
...
Scheduler ──生成 EOS──▶ DetokenizerManager ──[完成]──▶ 用户
```

DetokenizerManager 会将每个新的 Token（或几个 Token 一组）转换为文字，立即通过 HTTP 的 Server-Sent Events（SSE）推送给用户。

---

## 5. 分词的挑战

### 挑战 1：多语言支持

不同语言的分词难度不同：
- 英文有空格分隔单词，相对容易
- 中文没有空格，需要更智能的分割
- 混合语言（"这个model很好用"）更复杂

### 挑战 2：特殊标记

模型需要一些特殊标记来理解文本结构：
- `<|begin_of_text|>` — 文本开始
- `<|end_of_text|>` — 文本结束（也叫 EOS）
- `<|user|>` — 用户消息开始
- `<|assistant|>` — 助手回答开始

这些特殊标记不会显示给用户，但对模型理解文本至关重要。

### 挑战 3：Token 边界

有时候，一个 Token 只是一个字的一部分，反分词时需要正确拼接：

```
生成过程中：
Token#1 → "hel"  （半个单词）
Token#2 → "lo"   （另一半）
拼接后 → "hello" ✓
```

DetokenizerManager 需要处理这种"跨 Token 边界"的情况。

---

## 6. 下一步

- 分词之后的调度过程 → [调度系统](06-scheduling.md)
- 返回请求流程全貌 → [推理请求的旅程](04-request-journey.md)
- 返回目录 → [README](README.md)
