# 🚀 SGLang 推理引擎学习指南

> **写给零编程经验的你**：本指南用日常语言和生活比喻，带你理解 SGLang 如何让大语言模型（LLM）在服务器上高速运转。

---

## 📖 SGLang 是什么？

**一句话解释**：SGLang 是一个"大模型推理引擎"——就像汽车的发动机让汽车能跑起来，SGLang 让大语言模型（如 LLaMA、Qwen）能够快速响应用户的提问。

**日常比喻**：想象一家超级忙碌的餐厅 🍜
- **大语言模型** = 厨师的菜谱和烹饪技能（知识存储在 GPU 显存中）
- **SGLang** = 餐厅的整套运营系统（接单、排队、分配厨师、上菜）
- **用户请求** = 顾客的点单
- **推理结果** = 做好的菜品

没有 SGLang 这样的引擎，大模型就像一个只会做菜但不会接单、不会排队的厨师——有能力但无法服务大量顾客。

---

## 🗺️ 学习路线图

### 路线 A：快速入门（约 2 小时）

适合想快速了解全貌的读者：

| 顺序 | 文档 | 你将了解 |
|------|------|----------|
| 1 | [全景概览](01-overview.md) | SGLang 是什么、能做什么 |
| 2 | [架构设计](02-architecture.md) | 核心组件和它们的关系 |
| 3 | [一次推理请求的旅程](04-request-journey.md) | 从提问到回答的完整过程 |
| 4 | [术语表](10-glossary.md) | 遇到不懂的词随时查阅 |

### 路线 B：深入理解（约 12 小时）

适合想彻底弄懂原理的读者，按顺序阅读所有文档：

| 顺序 | 文档 | 核心主题 |
|------|------|----------|
| 1 | [全景概览](01-overview.md) | 项目全貌 |
| 2 | [架构设计](02-architecture.md) | 系统架构 |
| 3 | [服务器启动](03-server-startup.md) | 启动流程 |
| 4 | [推理请求旅程](04-request-journey.md) | 完整请求追踪 |
| 5 | [分词与反分词](05-tokenization.md) | Token 处理 |
| 6 | [调度系统](06-scheduling.md) | 批次调度 |
| 7 | [模型推理](07-model-inference.md) | GPU 计算 |
| 8 | [KV 缓存](08-kv-cache.md) | 缓存优化 |
| 9 | [采样与生成](09-sampling.md) | 文本生成策略 |
| 10 | [术语表](10-glossary.md) | 完整术语参考 |

### 路线 C：高级特性（约 6 小时）⭐ 新增

适合已掌握基础、想了解高级功能的读者：

| 顺序 | 文档 | 核心主题 |
|------|------|----------|
| 1 | [高级功能详解](11-advanced-features.md) | 结构化输出、函数调用、多模态、LoRA |
| 2 | [生产环境部署](12-production-deployment.md) | 监控指标、分布式配置、性能调优 |

---

## 🔤 核心术语速查

| 术语 | 英文 | 一句话解释 |
|------|------|-----------|
| 推理 | Inference | 让模型根据输入生成回答的过程（≠ 训练） |
| 分词 | Tokenization | 把文字切成模型能理解的小块（Token） |
| Token | Token | 文字的最小单位，类似汉字中的"字" |
| 预填充 | Prefill | 模型一次性"读完"你的问题 |
| 解码 | Decode | 模型一个字一个字地"写"回答 |
| KV 缓存 | KV Cache | 模型的"记忆草稿纸"，避免重复计算 |
| 张量并行 | Tensor Parallelism | 把一个大模型拆到多块 GPU 上运行 |
| 批处理 | Batching | 把多个请求打包一起处理，提高效率 |
| 调度 | Scheduling | 决定先处理谁、后处理谁的策略 |
| 采样 | Sampling | 从多个候选词中选出下一个词的方法 |

---

## 📁 源码注释文件索引

以下源码文件已添加详细中文注释，建议配合学习文档一起阅读：

### 核心推理流程（Phase 1）

| 推理流程顺序 | 文件路径 | 核心内容 |
|-------------|---------|----------|
| ⓪ 配置 | `python/sglang/srt/server_args.py` | 服务器启动参数 |
| ① 入口 | `python/sglang/srt/entrypoints/http_server.py` | HTTP 服务和 API 路由 |
| ② 引擎 | `python/sglang/srt/entrypoints/engine.py` | 子进程启动和组件初始化 |
| ③ 数据结构 | `python/sglang/srt/managers/io_struct.py` | 进程间通信的消息格式 |
| ④ 分词 | `python/sglang/srt/managers/tokenizer_manager.py` | 文本→Token 的转换 |
| ⑤ 调度 | `python/sglang/srt/managers/scheduler.py` | 请求的排队和批次调度 |
| ⑥ 批次 | `python/sglang/srt/managers/schedule_batch.py` | 批次数据结构管理 |
| ⑦ 并行 | `python/sglang/srt/managers/tp_worker.py` | 张量并行工作进程 |
| ⑧ 推理 | `python/sglang/srt/model_executor/model_runner.py` | GPU 上的模型前向计算 |
| ⑨ 反分词 | `python/sglang/srt/managers/detokenizer_manager.py` | Token→文本 + 流式输出 |

### 采样与缓存系统（Phase 2）

| 模块 | 文件路径 | 核心内容 | 关键比喻 |
|------|---------|----------|----------|
| 采样参数 | `python/sglang/srt/sampling/sampling_params.py` | 控制生成行为的参数（温度、top-p等） | "文本生成的控制面板" |
| 采样器 | `python/sglang/srt/layers/sampler.py` | 从 logits 采样 token | "决策轮盘" |
| RadixAttention | `python/sglang/srt/mem_cache/radix_cache.py` | 🔥 **SGLang 核心创新**：前缀自动共享 | "图书馆卡片目录" |
| 内存池 | `python/sglang/srt/mem_cache/memory_pool.py` | GPU 显存分页管理 | "操作系统虚拟内存" |

### 高级功能（Phase 3）

| 功能分类 | 文件路径 | 核心内容 | 关键比喻 |
|---------|---------|----------|----------|
| **结构化输出** | | | |
| 语法管理 | `python/sglang/srt/constrained/grammar_manager.py` | 语法编译缓存与同步 | "严格的编辑" |
| XGrammar后端 | `python/sglang/srt/constrained/xgrammar_backend.py` | FSM约束解码实现 | "交通灯系统" |
| **函数调用** | | | |
| 调用解析器 | `python/sglang/srt/function_call/function_call_parser.py` | 提取模型输出的工具调用 | "翻译器" |
| 格式检测器 | `python/sglang/srt/function_call/base_format_detector.py` | 识别不同模型的调用格式 | "方言识别器" |
| **多模态** | | | |
| 多模态工具 | `python/sglang/srt/multimodal/mm_utils.py` | 图像预处理和编码 | "图像翻译器" |
| ViT加速器 | `python/sglang/srt/multimodal/vit_cuda_graph_runner.py` | CUDA Graph 视觉编码优化 | "录制-回放系统" |
| **LoRA动态适配** | | | |
| LoRA管理器 | `python/sglang/srt/lora/lora_manager.py` | 多适配器并发服务 | "演员更衣室" |
| LoRA层 | `python/sglang/srt/lora/layers.py` | 低秩适配数学实现 | "给西装加补丁" |

### 生产环境部署（即将补充）

监控指标、分布式配置、性能调优等生产级特性，详见 [12-production-deployment.md](12-production-deployment.md)

---

## 🔄 核心推理流程一览

```
用户提问
  │
  ▼
┌─────────────────┐
│  HTTP Server     │  ← 接收请求（像餐厅前台）
│  (FastAPI)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TokenizerManager │  ← 分词（把文字切成 Token）
│  (分词器)        │
└────────┬────────┘
         │  ZMQ 消息
         ▼
┌─────────────────┐
│   Scheduler      │  ← 调度（决定处理顺序）
│  (调度器)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ModelRunner     │  ← GPU 推理（模型计算）
│  (模型运行器)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Detokenizer     │  ← 反分词（Token 变回文字）
│  (反分词器)      │
└────────┬────────┘
         │
         ▼
    返回回答给用户
```

---

## 💡 阅读建议

1. **不要害怕源码**：注释已经用大量比喻解释了每一步，读起来像故事书
2. **善用术语表**：遇到不懂的词，翻到 [术语表](10-glossary.md) 查一查
3. **跟着流程走**：理解推理流程比记住每个细节更重要
4. **动手试试**：如果有条件，启动 SGLang 服务，发送请求，观察日志

---

> 📝 **贡献者说明**：本学习指南和源码注释仅供学习参考，不影响任何代码逻辑。如有疏漏或错误，欢迎反馈。
