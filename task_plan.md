# SGLang 推理流程学习注释与文档项目

> 目标：为 SGLang 核心推理路径添加中文注释，并创建面向零编程经验用户的学习指南

## 项目概述

**SGLang** 是一个高性能 LLM 推理引擎。本项目将：
1. 为核心推理路径的源代码添加详细中文注释（借鉴 /github/example/llm.ts 的注释风格）
2. 创建 `learning-guide/` 文档目录，包含从入门到深入的系列学习文档

## 注释风格（借鉴 /github/example/llm.ts）

- **语言**：全部中文
- **文件头部**：用 `# ====` 分隔的大注释块，包含：
  - 这个文件是做什么的？（一句话概述）
  - 工作流程（简化版，步骤列表）
  - 关键概念解释（比喻+解释）
- **类和方法上方**：docstring 风格中文注释
- **代码内部**：`# ======== 第N步：做了什么 ========` 风格的分段注释
- **比喻说明**：大量使用日常生活比喻帮助理解
- **import 注释**：每个重要 import 都有简短说明

## 文档风格（借鉴 /github/example/learning-guide/）

- **文件夹**：`learning-guide/`
- **命名**：`数字序号-英文描述.md`（如 `01-overview.md`）
- **README.md**：目录索引 + 学习路线图 + 术语对照表
- **内容**：中文、ASCII 图表、代码引用（含文件路径和行号）、生活比喻、表格总结
- **面向受众**：零编程经验，每个技术术语都需解释

---

## Phase 1: 核心源码注释 [pending]

按推理请求的完整流程顺序注释以下文件：

### 1.1 服务器入口与配置 [pending]
- [ ] `python/sglang/srt/server_args.py` — 服务器启动参数
- [ ] `python/sglang/srt/entrypoints/http_server.py` — HTTP API 入口（FastAPI）
- [ ] `python/sglang/srt/entrypoints/engine.py` — 引擎入口（启动子进程）

### 1.2 请求处理管线 [pending]
- [ ] `python/sglang/srt/managers/io_struct.py` — 进程间通信数据结构
- [ ] `python/sglang/srt/managers/tokenizer_manager.py` — 分词管理器（主进程）
- [ ] `python/sglang/srt/managers/detokenizer_manager.py` — 反分词管理器

### 1.3 调度与推理 [pending]
- [ ] `python/sglang/srt/managers/scheduler.py` — 调度器（GPU 管理）
- [ ] `python/sglang/srt/managers/schedule_batch.py` — 批处理调度
- [ ] `python/sglang/srt/managers/tp_worker.py` — 张量并行工作器
- [ ] `python/sglang/srt/model_executor/model_runner.py` — 模型前向推理

---

## Phase 2: 学习文档 [pending]

### 文档目录结构
```
learning-guide/
├── README.md              — 目录索引 + 学习路线图
├── 01-overview.md         — SGLang 全景：是什么、能做什么
├── 02-architecture.md     — 架构设计：多进程架构与数据流
├── 03-server-startup.md   — 服务器启动：从命令行到就绪
├── 04-request-journey.md  — 一次推理请求的完整旅程
├── 05-tokenization.md     — 分词与反分词
├── 06-scheduling.md       — 调度系统：Prefill 与 Decode
├── 07-model-inference.md  — 模型推理：前向传播
├── 08-kv-cache.md         — KV 缓存与内存管理
├── 09-sampling.md         — 采样与文本生成
└── 10-glossary.md         — 术语表与概念索引
```

---

## SGLang 推理架构（发现）

```
用户请求 (HTTP/gRPC)
    │
    ▼
┌─────────────────────┐
│  HTTP Server        │  ← FastAPI, 路由请求
│  (http_server.py)   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  TokenizerManager   │  ← 主进程，分词 + 请求管理
│  (tokenizer_manager │
│   .py)              │     通过 ZMQ 发送给 Scheduler
└─────────────────────┘
    │ (ZMQ IPC)
    ▼
┌─────────────────────┐
│  Scheduler          │  ← 子进程，管理 GPU 工作器
│  (scheduler.py)     │     调度 Prefill/Decode 批次
│                     │     使用 continuous batching
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  TpWorker           │  ← 张量并行工作器
│  (tp_worker.py)     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  ModelRunner        │  ← 前向传播执行
│  (model_runner.py)  │     加载模型权重
│                     │     CUDA Graph 优化
└─────────────────────┘
    │
    ▼ (输出 token IDs)
┌─────────────────────┐
│  DetokenizerManager │  ← 子进程，token → 文本
│  (detokenizer_      │     流式输出
│   manager.py)       │
└─────────────────────┘
    │
    ▼
返回给用户
```

关键设计：
- HTTP Server + TokenizerManager 在主进程
- Scheduler 和 DetokenizerManager 是独立子进程
- 进程间通过 ZMQ 通信
- 入口命令：`sglang` CLI → `sglang.cli.main:main`

---

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 子代理 Read 工具调用失败 | 1 | 改为主进程直接读取文件 |
