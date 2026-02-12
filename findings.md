# SGLang 推理流程研究发现

## 1. 示例项目注释风格分析

### 代码注释（/github/example/llm.ts）
- **文件头部大注释块**：用 `// ====` 包裹，包含"这个文件做什么"、"工作流程"、"关键概念"
- **import 注释**：每个 import 都有一行简短中文说明
- **函数/方法注释**：`// ======== 第N步：xxx ========` 分段注释
- **行内注释**：关键逻辑前用 `//` 解释"为什么"
- **比喻**：大量使用日常生活比喻（寄快递、看电影、流水线等）
- **语言**：全中文

### 文档结构（/github/example/learning-guide/）
- **文件夹名**：`learning-guide/`
- **文件命名**：`数字-英文.md`（01-overview.md, 02-architecture.md...）
- **README.md**：含学习路线图、术语对照表、源码文件索引表
- **文档内容**：
  - 标题层级：`# 章名` → `## N.M 节名` → `### 子节`
  - 大量 ASCII 流程图
  - 代码引用含文件路径和行号
  - 表格总结
  - 每章有"本章小结"

## 2. SGLang 核心架构

### 入口点
- CLI 入口：`sglang` → `sglang.cli.main:main`（pyproject.toml 第157行）
- 服务器启动：`launch_server()` in `http_server.py:1819`
- 引擎启动：`_launch_subprocesses()` in `engine.py:979`

### 三大核心组件
1. **TokenizerManager**（`managers/tokenizer_manager.py:188`）
   - 运行在主进程
   - 负责文本分词（text → token IDs）
   - 通过 ZMQ 将请求发送给 Scheduler
   - 混入（Mixin）：TokenizerCommunicatorMixin, TokenizerManagerMultiItemMixin

2. **Scheduler**（`managers/scheduler.py:247`）
   - 运行在子进程
   - 管理 GPU 工作器（TpWorker）
   - 调度 prefill 和 decode 批次
   - 混入众多：OutputProcessor, UpdateWeights, Profiler, Metrics, Disaggregation, PP, DPAttn, Dllm
   - 有两种事件循环：`event_loop_normal`（:1066）和 `event_loop_overlap`（:1093）

3. **DetokenizerManager**（`managers/detokenizer_manager.py`）
   - 运行在子进程
   - 将 token IDs 转回文本
   - 处理流式输出

### 模型执行层
- **TpWorker**（`managers/tp_worker.py:59`）：张量并行工作器基类
- **ModelRunner**（`model_executor/model_runner.py:274`）：
  - 负责模型前向传播
  - 管理 CUDA Graph
  - 方法：`forward_decode`（:2226）、`forward_extend`（:2249）、`forward`（:2329）

### 进程间通信
- 使用 ZMQ 库进行 IPC
- 每个进程使用不同端口（PortArgs）

### 关键数据结构
- `GenerateReqInput`：生成请求输入
- `BatchTokenizedGenerateReqInput`：批量分词后的生成请求
- `ForwardBatch`：前向传播批次
- `ScheduleBatch`：调度批次
- `ModelRunnerOutput`（:268）：模型输出

## 3. SGLang 目录结构

```
python/sglang/srt/
├── entrypoints/        ← 入口点（HTTP server, Engine, gRPC）
├── managers/           ← 管理器（Tokenizer, Scheduler, Detokenizer）
├── model_executor/     ← 模型执行（ModelRunner, CUDA Graph）
├── models/             ← 具体模型实现（Llama, Qwen, GPT 等）
├── layers/             ← 模型层（attention, MoE, normalization）
├── mem_cache/          ← 内存缓存（KV cache, token pool）
├── sampling/           ← 采样策略
├── configs/            ← 配置
├── distributed/        ← 分布式（TP, PP, DP）
├── lora/               ← LoRA 适配器
├── speculative/        ← 投机采样
├── multimodal/         ← 多模态支持
└── utils/              ← 工具函数
```

## 4. Python 注释格式适配

原始 TypeScript 注释使用 `// ====` 和 `//`。
Python 版本适配：
- 文件头：使用三引号 docstring + `# ====` 注释块
- 函数/方法：Python docstring `""" ... """`
- 行内：`# ` 注释
- 分段标记：`# ======== 第N步：xxx ========`
