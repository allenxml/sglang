# 📋 调度系统

> **目标**：理解 Scheduler 如何管理多个请求、什么是 Continuous Batching、Prefill 和 Decode 的区别。

---

## 1. 为什么需要调度？

### 比喻：餐厅经理的难题 🍳

想象你是一家繁忙餐厅的经理，面对这些挑战：
- 20 桌客人同时在等菜
- 只有 4 个灶台（GPU 资源有限）
- 有的菜要炒（Prefill），有的菜在炖（Decode）
- 有的客人急着走（低延迟要求），有的客人不急（高吞吐要求）

你需要决定：**先做谁的菜？一次做几份？怎么安排灶台？**

这就是 Scheduler（调度器）要解决的问题。

---

## 2. Prefill vs Decode

这是理解调度的关键——推理分为两个截然不同的阶段：

### Prefill（预填充）—— "通读题目"

```
输入: "什么是人工智能？请详细解释。"
      [Token1, Token2, Token3, ..., Token15]
                    ↓
           一次性全部处理（并行）
                    ↓
输出: KV Cache（所有 Token 的中间结果）+ 第一个生成 Token
```

**特点**：
- 处理所有输入 Token（可能几百上千个）
- **计算密集**：大量矩阵乘法
- 只需执行一次
- 比喻：考试时先通读整道题目

### Decode（解码）—— "逐字写答案"

```
输入: 上一步生成的 Token + KV Cache
                    ↓
          只处理一个新 Token
                    ↓
输出: 下一个 Token
```

**特点**：
- 每次只处理一个 Token
- **内存密集**：需要读取大量 KV Cache
- 需要重复很多次（每个输出 Token 一次）
- 比喻：读完题后一个字一个字写答案

### 两阶段对比

| | Prefill | Decode |
|--|---------|--------|
| 处理数量 | 全部输入 Token（几百个） | 1 个 Token |
| 执行次数 | 1 次 | N 次（N = 输出长度） |
| 瓶颈 | 计算（GPU 算力） | 内存（显存带宽） |
| 耗时 | 几十到几百毫秒 | 每次几十毫秒 |

> 💡 **关键洞察**：Prefill 和 Decode 的计算特性完全不同，但可以同时在 GPU 上执行（混合批处理），这就是 Continuous Batching 的基础。

---

## 3. Continuous Batching（连续批处理）

### 传统方式：Static Batching

```
时间 →
批次1: [请求A(Prefill)] [请求A(Decode)] [请求A(Decode)] [请求A(完成)]
       [请求B(Prefill)] [请求B(Decode)] [请求B(Decode)] [请求B(完成)]
                                                         ↑
                                                    等 A 和 B 都完成
                                                    才开始批次 2
批次2: [请求C(Prefill)] [请求C(Decode)] ...
       [请求D(Prefill)] [请求D(Decode)] ...
```

**问题**：请求 A 如果只需生成 10 个字，请求 B 需要生成 100 个字，A 完成后 GPU 就空闲了，浪费资源。

### SGLang 的方式：Continuous Batching

```
时间 →
步骤1: [A-Prefill] [B-Prefill]          ← 同时处理两个新请求
步骤2: [A-Decode]  [B-Decode]           ← 同时生成下一个字
步骤3: [A-Decode]  [B-Decode]  [C-Prefill]  ← C 随时加入！
步骤4: [A-完成✓]   [B-Decode]  [C-Decode]   ← A 完成，位置空出
步骤5: [D-Prefill] [B-Decode]  [C-Decode]   ← D 立刻补上！
```

**优势**：
- 不需要等整个批次完成
- 新请求随时加入
- GPU 始终满载工作
- 大幅提高吞吐量

> 💡 **比喻**：传统方式像旅行团——必须等所有人都上车才出发。Continuous Batching 像公交车——到站就上人，坐满就走，随上随走。

---

## 4. Scheduler 的工作流程

**文件**：`python/sglang/srt/managers/scheduler.py`

### 核心事件循环

Scheduler 运行一个无限循环（event loop），每一轮做以下事情：

```
┌─────────────────────────────────────┐
│          Scheduler 事件循环          │
│                                     │
│  1. 接收新请求（从 ZMQ）            │
│     └── 加入 waiting_queue          │
│                                     │
│  2. 获取下一个批次                   │
│     ├── 检查资源（显存、缓存空间）   │
│     ├── 选择 Decode 请求（优先）     │
│     └── 选择 Prefill 请求           │
│                                     │
│  3. 执行批次                        │
│     └── 调用 ModelRunner 前向传播    │
│                                     │
│  4. 处理结果                        │
│     ├── 采样得到新 Token             │
│     ├── 检查是否完成                 │
│     └── 发送完成的结果               │
│                                     │
│  ──── 回到第 1 步 ────              │
└─────────────────────────────────────┘
```

### 两个队列

```
waiting_queue（等待队列）:
  ┌──────┬──────┬──────┐
  │ Req5 │ Req6 │ Req7 │  ← 新请求在这里排队
  └──────┴──────┴──────┘

running_batch（运行批次）:
  ┌──────┬──────┬──────┬──────┐
  │ Req1 │ Req2 │ Req3 │ Req4 │  ← 正在 GPU 上运行的请求
  └──────┴──────┴──────┴──────┘
```

### 调度策略

Scheduler 需要做一些关键决策：

**决策 1：这一轮处理什么？**
- 如果有正在 Decode 的请求 → 优先继续 Decode（不能让生成中的请求等太久）
- 如果 GPU 有余力 → 同时加入新请求做 Prefill

**决策 2：可以加多少请求？**
- 检查 KV Cache 是否有足够空间
- 检查是否超过最大批次大小
- 为正在运行的请求保留足够的 Decode 空间

**决策 3：显存不够怎么办？**
- 暂时"搁置"一些请求（swap/evict）
- 等其他请求完成、释放 KV Cache 空间后再继续

---

## 5. 批次数据结构

**文件**：`python/sglang/srt/managers/schedule_batch.py`

### Req（单个请求）

```python
class Req:
    rid: str                    # 请求唯一 ID
    origin_input_ids: List[int] # 原始输入 Token ID
    output_ids: List[int]       # 已生成的 Token ID
    sampling_params: ...        # 采样参数（temperature 等）
```

### ScheduleBatch（一个批次）

```python
class ScheduleBatch:
    reqs: List[Req]             # 这个批次包含的所有请求
    batch_size: int             # 批次大小
    # ... 用于 GPU 计算的各种张量
```

---

## 6. 下一步

- 了解 GPU 上如何执行推理 → [模型推理](07-model-inference.md)
- 了解 KV Cache 的原理 → [KV 缓存](08-kv-cache.md)
- 返回目录 → [README](README.md)
