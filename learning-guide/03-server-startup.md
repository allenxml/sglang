# 🚦 服务器启动流程

> **目标**：读完本文，你将了解从运行一行命令到服务就绪，SGLang 后台发生了什么。

---

## 1. 启动命令

启动一个 SGLang 服务，只需一行命令：

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3-8B --port 8000
```

看起来很简单，但背后发生了一连串精密的操作——就像你按下汽车启动键，发动机、电路、油路、空调都要依次启动。

---

## 2. 启动的三个阶段

### 阶段一：解析参数（ServerArgs）

**做什么**：读取命令行参数，确定"怎么启动"

**文件**：`python/sglang/srt/server_args.py`

```
命令行参数 → ServerArgs 对象
```

常用参数一览：

| 参数 | 含义 | 比喻 |
|------|------|------|
| `--model-path` | 模型路径或 HuggingFace 名称 | 告诉厨师用哪本菜谱 |
| `--port` | HTTP 服务端口 | 餐厅的门牌号 |
| `--tp-size` | 张量并行数量（用几块 GPU） | 安排几个厨师一起做 |
| `--mem-fraction-static` | 显存中固定分配给 KV Cache 的比例 | 预留多大的备菜台 |
| `--context-length` | 最大上下文长度 | 一次能读多长的题目 |
| `--max-running-requests` | 最大并发请求数 | 餐厅最多同时接待几桌 |

### 阶段二：启动子进程（Engine）

**做什么**：创建必要的子进程和组件

**文件**：`python/sglang/srt/entrypoints/engine.py`

启动顺序：

```
1. 创建 TokenizerManager（主进程内）
   └── 加载分词器（Tokenizer），准备处理文本

2. 创建 DetokenizerManager（主进程内）
   └── 准备把 Token 转回文本

3. 启动 Scheduler 进程（新进程）
   ├── 创建 Scheduler 调度器
   ├── 创建 TpWorker（张量并行工作器）
   ├── 加载模型权重到 GPU
   └── 分配 KV Cache 显存

4. 建立 ZMQ 通信通道
   └── TokenizerManager ↔ Scheduler ↔ DetokenizerManager
```

> 💡 **比喻**：就像餐厅开业前的准备工作——先装修前台（TokenizerManager），再建后厨（Scheduler + ModelRunner），最后架好对讲机（ZMQ），确保前后厨能通话。

### 阶段三：启动 HTTP 服务

**做什么**：启动 FastAPI Web 服务器，开始接收请求

**文件**：`python/sglang/srt/entrypoints/http_server.py`

```
FastAPI 应用注册路由：
├── POST /v1/chat/completions  → OpenAI 兼容的对话接口
├── POST /v1/completions       → OpenAI 兼容的补全接口
├── POST /generate             → SGLang 原生接口
├── GET  /health               → 健康检查
├── GET  /v1/models            → 模型列表
└── ...更多管理接口
```

当你看到日志输出 `"The server is fired up and ready to roll!"` 时，服务就可以接收请求了。

---

## 3. 启动时的关键步骤：模型加载

在整个启动过程中，最耗时的是"模型加载"——把模型的权重参数从磁盘读入 GPU 显存。

```
磁盘上的模型文件（几 GB 到几百 GB）
        │
        ▼
    读入内存（CPU RAM）
        │
        ▼
    转移到 GPU 显存（VRAM）
        │
        ▼
    模型准备就绪 ✅
```

> 💡 **比喻**：这就像厨师开工前要把所有食材从冰库搬到操作台上——模型文件就是"食材"，GPU 显存就是"操作台"。食材越多（模型越大），搬运时间越长。

**常见模型大小参考**：

| 模型 | 参数量 | 大约显存需求 | 加载时间 |
|------|--------|------------|---------|
| Llama-3-8B | 80 亿 | ~16 GB | ~30 秒 |
| Llama-3-70B | 700 亿 | ~140 GB | ~2 分钟 |
| Qwen-72B | 720 亿 | ~144 GB | ~2 分钟 |

---

## 4. 启动流程图

```
python -m sglang.launch_server --model-path ... --port 8000
    │
    ▼
┌──────────────────────┐
│ 1. 解析命令行参数     │
│    → ServerArgs 对象   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 2. 初始化引擎 Engine  │
│    _launch_subprocesses() │
└──────────┬───────────┘
           │
    ┌──────┴──────────────────┐
    │                         │
    ▼                         ▼
┌────────────┐  ┌──────────────────────┐
│ 主进程      │  │ Scheduler 子进程      │
│            │  │                      │
│ Tokenizer  │  │ - 初始化 Scheduler   │
│ Manager    │  │ - 创建 TpWorker      │
│            │  │ - 加载模型到 GPU  🔥  │
│ Detokenizer│  │ - 分配 KV Cache 显存  │
│ Manager    │  │ - 预热 CUDA Graph    │
│            │  │                      │
└──────┬─────┘  └──────────┬───────────┘
       │                   │
       │    ZMQ 通道建立    │
       └───────────────────┘
           │
           ▼
┌──────────────────────┐
│ 3. 启动 HTTP 服务     │
│    uvicorn + FastAPI  │
│    监听端口 8000      │
└──────────────────────┘
           │
           ▼
    ✅ 服务就绪！
    "The server is fired up and ready to roll!"
```

---

## 5. 下一步

- 服务启动后，一个请求是怎么被处理的？→ [推理请求的旅程](04-request-journey.md)
- 想了解 ServerArgs 的详细参数 → 查看源码注释：`python/sglang/srt/server_args.py`
- 返回目录 → [README](README.md)
