# 进度记录

## Session 1 — 2026-02-11

### 已完成
- [x] 阅读 /github/example/llm.ts 注释风格（641行，全中文详细注释）
- [x] 阅读 /github/example/learning-guide/ 文档结构
  - README.md：学习路线图 + 术语表 + 源码索引
  - 01-overview.md：全景文档样例
  - 09-request-trace.md：请求追踪文档样例（900行）
- [x] 探索 sglang 项目目录结构
- [x] 确认入口点：CLI → launch_server → _launch_subprocesses
- [x] 阅读核心文件头部：http_server.py, engine.py, tokenizer_manager.py, scheduler.py, model_runner.py, tp_worker.py
- [x] 确认推理流程链路：HTTP → TokenizerManager → Scheduler → TpWorker → ModelRunner → Detokenizer
- [x] 创建 task_plan.md、findings.md、progress.md

### 当前阶段
- Phase 1: 核心源码注释（待开始）
- Phase 2: 学习文档（待开始）

### 文件修改记录
| 文件 | 操作 | 状态 |
|------|------|------|
| task_plan.md | 创建 | 完成 |
| findings.md | 创建 | 完成 |
| progress.md | 创建 | 完成 |
