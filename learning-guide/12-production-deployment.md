# 🚀 生产环境部署指南

> **适合读者**：已理解 SGLang 核心推理流程，准备在生产环境部署大模型服务的工程师。

---

## 📚 本章内容概览

| 模块 | 典型应用场景 | 核心价值 |
|------|-------------|---------|
| 监控指标 | 服务健康监测 | 实时掌握系统状态，快速定位问题 |
| 分布式部署 | 大规模服务 | 单机突破，横向扩展吞吐量 |
| 性能调优 | 降低延迟/提高吞吐 | 优化资源利用率，降低成本 |
| 配置管理 | 多模型服务 | 灵活适配不同模型架构 |
| **PD 分离部署** ⭐ | **流式输出延迟敏感** | **5-10x 吞吐提升，消除 Prefill 中断** |
| **AI 网关路由** ⭐ | **成本优化** | **降低 85% 成本，保持 95% 质量** |
| 最佳实践 | 生产清单 | 监控、资源规划、故障预案 |
| 性能基准 | 硬件选型 | 典型配置的吞吐量/延迟数据 |
| 故障排查 | 问题诊断 | OOM、延迟、吞吐量问题解决方案 |

---

## 1️⃣ 监控与指标系统（Metrics & Monitoring）

### 🎯 解决什么问题？

**问题**：生产环境如何知道服务是否健康？性能是否下降？

**典型场景**：
- 📊 实时监控：当前有多少请求在处理？队列积压了多少？
- ⚠️ 告警触发：响应时间超过 2 秒，触发告警
- 📈 容量规划：根据历史数据预测需要多少 GPU

### 💡 解决方案：Prometheus 指标体系

**架构**：
```
SGLang 服务
    ↓
[RequestMetricsExporter] 指标导出器
    ├─ 收集各组件指标
    │   ├─ Scheduler: 队列长度、批次大小
    │   ├─ ModelRunner: GPU利用率、推理延迟
    │   └─ TokenizerManager: 分词速度
    ↓
[Prometheus] 时序数据库
    ├─ 存储历史指标（默认保留15天）
    └─ 提供查询接口（PromQL）
    ↓
[Grafana] 可视化面板
    ├─ 实时仪表盘
    ├─ 告警规则配置
    └─ 趋势分析图表
```

### 📊 核心指标类别

#### A. 吞吐量指标（Throughput Metrics）

| 指标名 | 含义 | 健康范围 | 告警阈值 |
|--------|------|----------|----------|
| `requests_total` | 累计请求数 | 持续增长 | - |
| `tokens_generated_total` | 累计生成token数 | 持续增长 | - |
| `throughput_tps` | 每秒生成token数（TPS） | 500-2000 | <100 |
| `batch_size_avg` | 平均批次大小 | 8-32 | <4 |

**比喻理解**：吞吐量指标就像餐厅的"上菜速度"
- `requests_total` = 总共服务了多少桌客人
- `throughput_tps` = 每分钟出菜数量
- `batch_size_avg` = 平均每次端出去几盘菜

#### B. 延迟指标（Latency Metrics）

| 指标名 | 含义 | P50目标 | P99目标 |
|--------|------|---------|---------|
| `prefill_latency_ms` | 预填充延迟（首token时间） | <200ms | <500ms |
| `decode_latency_ms` | 解码延迟（每token时间） | <50ms | <100ms |
| `e2e_latency_ms` | 端到端延迟（总时间） | <2000ms | <5000ms |
| `queue_wait_ms` | 队列等待时间 | <100ms | <500ms |

**关键术语**：
- **P50（中位数）**：50% 的请求延迟低于此值
- **P99**：99% 的请求延迟低于此值（更能反映用户体验）

**示例场景**：
```
请求A的旅程：
├─ 0ms: 到达
├─ 50ms: 队列等待 (queue_wait_ms)
├─ 250ms: 预填充完成 (prefill_latency_ms = 200ms)
├─ 2050ms: 生成100个token (decode_latency_ms = 18ms/token)
└─ 2050ms: 返回用户 (e2e_latency_ms = 2050ms)
```

#### C. 资源利用率指标（Resource Utilization）

| 指标名 | 含义 | 健康范围 | 告警阈值 |
|--------|------|----------|----------|
| `gpu_memory_used_gb` | GPU显存占用 | 60-80% | >90% |
| `kv_cache_usage_ratio` | KV缓存利用率 | 50-70% | >85% |
| `radix_cache_hit_rate` | Radix缓存命中率 | >30% | <10% |
| `running_requests` | 当前处理的请求数 | 10-50 | >100 |

**比喻理解**：资源利用率就像餐厅的"桌位占用率"
- 太低（<50%）= 资源浪费，成本高
- 合理（60-80%）= 高效运转
- 太高（>90%）= 即将饱和，需要扩容

### 🔧 指标导出实现

**关键代码流程**：
```python
[RequestMetricsExporter] 核心类
    │
    ├─ __init__(): 初始化所有指标对象
    │   ├─ Counter: 累计型指标（如 requests_total）
    │   ├─ Gauge: 快照型指标（如 running_requests）
    │   └─ Histogram: 分布型指标（如 latency_ms）
    │
    ├─ export_request_metrics(): 导出单次请求指标
    │   ├─ 记录延迟（e2e、prefill、decode）
    │   ├─ 记录生成token数
    │   └─ 更新吞吐量
    │
    └─ export_scheduler_metrics(): 导出调度器指标
        ├─ 当前批次大小
        ├─ 队列长度
        └─ GPU利用率
```

**Prometheus 指标格式示例**：
```
# HELP sglang_requests_total Total number of requests
# TYPE sglang_requests_total counter
sglang_requests_total 12345

# HELP sglang_e2e_latency_ms End-to-end latency in milliseconds
# TYPE sglang_e2e_latency_ms histogram
sglang_e2e_latency_ms_bucket{le="100"} 230
sglang_e2e_latency_ms_bucket{le="500"} 890
sglang_e2e_latency_ms_bucket{le="1000"} 1200
sglang_e2e_latency_ms_sum 3450000
sglang_e2e_latency_ms_count 1500
```

### 📝 关键代码文件

| 文件 | 核心职责 |
|------|---------|
| `request_metrics_exporter.py` | Prometheus 指标定义和导出 |
| `scheduler_metrics_mixin.py` | Scheduler 的指标收集逻辑 |
| `scheduler_profiler_mixin.py` | 性能分析器（详细时间分解） |

---

## 2️⃣ 分布式部署配置（Distributed Deployment）

### 🎯 解决什么问题？

**问题**：单机 GPU 无法满足大模型或高并发需求

**典型场景**：
- **大模型**：LLaMA-405B 需要 8× A100 80GB
- **高吞吐**：单机 1000 TPS，需要扩展到 10000 TPS
- **低延迟**：通过并行减少推理时间

### 💡 解决方案：三种并行策略

| 并行类型 | 适用场景 | 优势 | 劣势 |
|---------|---------|------|------|
| **Tensor Parallelism (TP)** | 超大模型（单卡放不下） | 延迟最低 | GPU间通信开销大 |
| **Data Parallelism (DP)** | 高吞吐需求 | 扩展性好，无通信开销 | 显存重复，成本高 |
| **Pipeline Parallelism (PP)** | 极大模型（TP不够用） | 显存需求小 | 流水线气泡，效率低 |

### 🔧 A. Data Parallelism（数据并行）

**核心思想**：多个副本独立处理不同请求

**架构比喻**：像多家独立餐厅
- 每家餐厅（DP节点）有完整的厨房设备（模型副本）
- 客人（请求）随机分配到不同餐厅
- 餐厅之间互不干扰

**实现流程**：
```
[DataParallelController] 控制器
    │
    ├─ 初始化 N 个 Worker 副本
    │   Worker-0: 模型副本 @ GPU 0-3 (TP=4)
    │   Worker-1: 模型副本 @ GPU 4-7 (TP=4)
    │   Worker-2: 模型副本 @ GPU 8-11 (TP=4)
    │
    ├─ 请求路由策略
    │   ├─ Round-Robin: 轮流分配
    │   ├─ Least-Load: 分配给最空闲的Worker
    │   └─ Sticky-Session: 同一用户固定Worker（利用缓存）
    │
    └─ 健康检查
        ├─ 定期ping各Worker
        └─ 故障Worker自动剔除
```

**配置示例**：
```bash
# 启动 DP=3, TP=4 的分布式服务（需要12张GPU）
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B \
  --dp-size 3 \       # 3个数据并行副本
  --tp-size 4 \       # 每个副本内4卡张量并行
  --port 30000
```

**性能特点**：
- **吞吐提升**：理论上线性扩展（DP=3 → 3倍吞吐）
- **延迟不变**：单个请求的处理时间不变
- **显存成本**：N倍显存占用（每个副本都是完整模型）

**适用场景**：
- ✅ 高并发在线服务（QPS > 1000）
- ✅ 模型不太大（单副本能用TP放下）
- ❌ 显存受限场景（成本敏感）

### 🔧 B. Pipeline Parallelism（流水线并行）

**核心思想**：模型按层切分到不同GPU，像流水线传递数据

**架构比喻**：像汽车组装流水线
- 第1站：安装底盘（Layer 0-10）
- 第2站：安装车身（Layer 11-20）
- 第3站：安装内饰（Layer 21-32）
- 一辆车要经过所有站点才能完成

**实现流程**：
```
输入 Batch
    ↓
[GPU 0] Layer 0-10 → 中间结果1
    ↓ (通过PCIe/NVLink传输)
[GPU 1] Layer 11-20 → 中间结果2
    ↓
[GPU 2] Layer 21-32 → 最终输出
```

**流水线气泡问题**：
```
时间轴：
GPU-0: [Batch1] [Batch2] [Batch3] [空闲] [空闲]
GPU-1: [空闲] [Batch1] [Batch2] [Batch3] [空闲]
GPU-2: [空闲] [空闲] [Batch1] [Batch2] [Batch3]

问题：GPU-0 处理 Batch1 时，GPU-1 和 GPU-2 空闲（气泡）
解决：Micro-batching（把 Batch 切分成更小的 micro-batch）
```

**配置示例**：
```bash
# 启动 PP=4 的流水线并行（需要4张GPU）
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-405B \
  --pp-size 4 \       # 4级流水线
  --port 30000
```

**性能特点**：
- **显存节省**：每张GPU只存1/N的模型层
- **延迟增加**：需要多次GPU间传输
- **吞吐受限**：流水线气泡导致效率<100%

**适用场景**：
- ✅ 超大模型（405B+）
- ✅ 显存极度受限
- ❌ 低延迟要求场景

### 🔧 C. 混合并行（TP + DP + PP）

**实际生产配置示例**：
```bash
# LLaMA-405B 高吞吐服务配置（需要24张A100）
# - TP=4: 单个副本4卡张量并行（降低延迟）
# - PP=2: 流水线2级（显存不够时启用）
# - DP=3: 3个副本（提高吞吐）
# 总GPU数 = TP × PP × DP = 4 × 2 × 3 = 24

python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-405B \
  --tp-size 4 \
  --pp-size 2 \
  --dp-size 3 \
  --port 30000
```

**决策树**：
```
需要部署的模型大小？
    ├─ <70B → TP=1 或 TP=2（单卡或双卡）
    │         高吞吐？DP=4-8
    │
    ├─ 70B-200B → TP=4（4卡张量并行）
    │             高吞吐？DP=2-4
    │
    └─ >200B → TP=8, PP=2-4（流水线 + 张量并行）
               高吞吐？DP=2
```

### 📝 关键代码文件

| 文件 | 核心职责 |
|------|---------|
| `data_parallel_controller.py` | DP副本管理、请求路由 |
| `scheduler_dp_attn_mixin.py` | DP场景下的注意力优化 |
| `scheduler_pp_mixin.py` | PP流水线调度逻辑 |

---

## 3️⃣ 性能调优（Performance Tuning）

### 🎯 解决什么问题？

**问题**：如何在有限资源下最大化性能？

**核心矛盾**：
- **吞吐 vs 延迟**：大批次提高吞吐但增加延迟
- **公平 vs 效率**：FCFS 公平但可能饿死小请求
- **缓存 vs 显存**：更大缓存提高命中率但占用显存

### 🔧 A. 调度策略（Scheduling Policies）

#### 策略对比表

| 策略 | 描述 | 优势 | 劣势 | 适用场景 |
|------|------|------|------|----------|
| **FCFS** | 先来先服务 | 公平，实现简单 | 大请求阻塞小请求 | 请求大小均匀 |
| **SJF** | 短任务优先 | 平均等待时间短 | 长任务可能饿死 | 混合负载 |
| **Priority** | 优先级队列 | 重要请求优先 | 需要额外优先级信息 | 多租户SaaS |
| **LJF** | 长任务优先 | 减少碎片 | 短任务延迟高 | 批处理场景 |

#### 调度策略实现

**代码结构**：
```python
[SchedulePolicy] 基类
    │
    ├─ FCFSPolicy: 按到达时间排序
    │   get_priority(req) → req.arrival_time
    │
    ├─ SJFPolicy: 按预估长度排序
    │   get_priority(req) → req.estimated_tokens
    │
    └─ PriorityPolicy: 按用户指定优先级
        get_priority(req) → req.priority_level
```

**配置示例**：
```bash
# 使用 SJF 策略
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B \
  --schedule-policy sjf \
  --port 30000
```

### 🔧 B. Prefill 延迟控制（Prefill Delayer）

**问题**：Prefill 阶段计算量大，可能阻塞 Decode

**比喻**：餐厅厨房的"备菜"与"炒菜"
- **Prefill** = 备菜（切菜、腌制）：耗时长但一次性
- **Decode** = 炒菜（一盘接一盘上菜）：快但需要持续输出
- **问题**：如果一直在备菜，已经在炒的菜会冷掉（用户看到的流式输出卡顿）

**解决方案：PrefillDelayer**

```
[调度器决策逻辑]
    │
    ├─ 检查当前批次中的 Decode 请求数
    │   ├─ 如果 Decode 请求 > 10
    │   │   → 延迟新的 Prefill 请求（避免阻塞）
    │   └─ 如果 Decode 请求 < 5
    │       → 允许 Prefill 请求加入
    │
    └─ 动态调整 Prefill 批次大小
        ├─ GPU 利用率 < 70% → 增加 Prefill 批次
        └─ GPU 利用率 > 90% → 减少 Prefill 批次
```

**配置参数**：
```python
--schedule-conservativeness 0.0-1.0
# 0.0: 激进（优先吞吐，可能增加延迟）
# 1.0: 保守（优先延迟，可能降低吞吐）
# 默认: 0.5（平衡）
```

**实际效果**：
```
无 Prefill 控制:
Decode 请求延迟: P50=80ms, P99=500ms ❌ (高方差)

有 Prefill 控制:
Decode 请求延迟: P50=60ms, P99=120ms ✅ (稳定)
```

### 🔧 C. 性能分析器（Profiler）

**核心功能**：细粒度时间分解

**Profiler 输出示例**：
```
========== Request #12345 Profiling ==========
Total Latency: 2.34s
  ├─ Queue Wait: 0.15s (6.4%)
  ├─ Prefill Phase: 0.32s (13.7%)
  │   ├─ Token Encoding: 0.05s
  │   ├─ GPU Compute: 0.22s
  │   └─ KV Cache Write: 0.05s
  ├─ Decode Phase: 1.80s (76.9%)
  │   ├─ GPU Compute: 1.50s
  │   ├─ Sampling: 0.10s
  │   └─ Detokenization: 0.20s
  └─ Network Transmission: 0.07s (3.0%)

Bottleneck: Decode GPU Compute (1.50s)
Recommendation: Consider TP=2 for lower latency
```

**使用方式**：
```bash
# 启用性能分析（会增加 5-10% 开销）
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B \
  --enable-profiler \
  --port 30000
```

### 📝 关键代码文件

| 文件 | 核心职责 |
|------|---------|
| `schedule_policy.py` | 调度策略实现（FCFS/SJF/Priority） |
| `prefill_delayer.py` | Prefill 延迟控制逻辑 |
| `scheduler_profiler_mixin.py` | 详细性能分析 |
| `scheduler_metrics_mixin.py` | 调度器指标收集 |

---

## 4️⃣ 配置管理（Configuration Management）

### 🎯 解决什么问题？

**问题**：如何灵活适配不同模型架构和硬件环境？

**挑战**：
- 不同模型架构差异大（LLaMA vs Qwen vs DeepSeek）
- 不同硬件能力不同（A100 vs H100 vs L40）
- 不同量化格式（FP16 vs INT8 vs INT4）

### 🔧 A. 模型加载配置（LoadConfig）

**核心职责**：定义如何加载模型权重

**配置流程**：
```
模型路径 (model_path)
    ↓
[load_config.py] 加载配置解析
    ├─ 检测模型格式
    │   ├─ HuggingFace 格式 (.safetensors)
    │   ├─ GGUF 格式 (.gguf)
    │   └─ vLLM 格式 (.bin)
    │
    ├─ 量化配置
    │   ├─ load_format: "auto", "fp16", "int8", "int4"
    │   ├─ quantization: "awq", "gptq", "bitsandbytes"
    │   └─ quantization_param_path: 量化配置文件
    │
    └─ 分布式配置
        ├─ tp_size: 张量并行大小
        ├─ pp_size: 流水线并行大小
        └─ model_loader_extra_config: 额外加载参数
```

**配置示例**：
```bash
# 加载 AWQ 量化的 LLaMA 模型
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B-Instruct-AWQ \
  --load-format awq \
  --quantization awq \
  --tp-size 2 \
  --port 30000
```

### 🔧 B. 模型架构配置（ModelConfig）

**核心职责**：定义模型的数学架构参数

**关键参数**：
```python
ModelConfig:
    ├─ hidden_size: 隐藏层维度（如 4096）
    ├─ num_hidden_layers: Transformer 层数（如 32）
    ├─ num_attention_heads: 注意力头数（如 32）
    ├─ vocab_size: 词汇表大小（如 128256）
    ├─ max_position_embeddings: 最大上下文长度（如 131072）
    ├─ rope_theta: RoPE 位置编码参数
    └─ sliding_window: 滑动窗口大小（Mistral 专用）
```

**自动检测流程**：
```python
# SGLang 会自动从 config.json 读取配置
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 8192,
  "num_hidden_layers": 80,
  "num_attention_heads": 64,
  "vocab_size": 128256,
  "rope_theta": 500000.0,  # LLaMA 3.1 特有
  ...
}

→ ModelConfig 自动解析并验证参数合法性
```

**特殊配置处理**：
```python
# Qwen2-VL 的特殊配置
if model_type == "qwen2_vl":
    config.vision_config = {
        "hidden_size": 1280,
        "image_size": 980,
        "patch_size": 14,
        ...
    }

# Mistral 的滑动窗口注意力
if model_type == "mistral":
    config.sliding_window = 4096
```

### 📝 关键代码文件

| 文件 | 核心职责 |
|------|---------|
| `configs/load_config.py` | 模型权重加载配置 |
| `configs/model_config.py` | 模型架构参数定义 |

---

## 5️⃣ PD 分离部署架构（Production PD Disaggregation）

### 🎯 什么是 PD 分离？

**核心思想**：将 LLM 推理的两个阶段分离到独立的服务集群

```
传统统一调度：
[单一服务] → Prefill + Decode 混合批次 → 互相干扰，延迟不稳定

PD 分离调度：
[Prefill 集群] → 专注计算密集型任务（处理长输入）
        ↓ KV Cache 传输
[Decode 集群] → 专注延迟敏感型任务（生成 tokens）
```

**解决的问题**：
1. **Prefill 中断**：大批次 Prefill 阻塞 Decode，导致用户看到的流式输出卡顿
2. **DP 注意力不平衡**：数据并行场景下，不同 Worker 处理不同阶段，负载不均

### 📊 生产案例 1：Kimi K2 部署（128 H200 GPUs）

**来源**：[LMSYS 博客 - Kimi K2 大规模部署](https://lmsys.org/blog/2025-07-20-k2-large-scale-ep/)

#### 硬件配置
```
总GPU数：128 × H200 GPU
拓扑结构：1P1D（1个 Prefill 节点对应 1个 Decode 节点）
  ├─ Prefill: 4 nodes × 8 GPUs = 32 GPUs
  └─ Decode:  12 nodes × 8 GPUs = 96 GPUs

网络：NVLink + InfiniBand RDMA
云平台：NVIDIA DGX Cloud
```

#### 架构设计
```
                 ┌─────────────┐
                 │  Router     │
                 │  (负载均衡)  │
                 └──────┬──────┘
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
    ┌────────────┐          ┌────────────┐
    │ Prefill    │ KV传输   │  Decode    │
    │ 4 nodes    ├─────────►│ 12 nodes   │
    │ 32 H200    │  RDMA    │ 96 H200    │
    └────────────┘          └────────────┘
    计算密集型                延迟敏感型
    (2000 tokens输入)        (100 tokens输出)
```

#### 性能数据
| 指标 | Prefill 集群 | Decode 集群 | 总计 |
|------|-------------|------------|------|
| 吞吐量 | 224K tokens/s | 288K tokens/s | 512K tokens/s |
| 单节点吞吐 | 56K tokens/s/node | 24K tokens/s/node | - |
| 批次大小 | 动态调整 | 480（优化值） | - |
| 成本效率 | - | - | $0.21/百万 tokens |

**关键配置**：
```bash
# 通过 OME (Open Model Engine) 部署
helm upgrade --install ome-crd <OCI-registry>
helm upgrade --install ome <OCI-registry>

# 关键参数
--deepep-mode=low_latency
--ep-dispatch-algorithm=dynamic
--enable-eplb  # 专家并行负载均衡器
--enable-dp-attention
--moe-a2a-backend=deepep
```

#### 生产经验
1. **资源配比**：Prefill:Decode = 1:3（32:96 GPUs），Decode 占大头因为需要处理大量并发流式请求
2. **专家并行（EP）**：减少单 GPU 显存压力，扩展 KV Cache 池
3. **模型驱动部署**："模型应该驱动部署方案，而非反之"

---

### 📊 生产案例 2：DeepSeek 部署（96 H100 GPUs）

**来源**：[LMSYS 博客 - DeepSeek 大规模部署](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

#### 硬件配置
```
总GPU数：96 × H100 GPU (12 nodes × 8 GPUs)
网络：RDMA-capable InfiniBand
云平台：Atlas Cloud
```

#### 性能数据
| 指标 | 数值 |
|------|------|
| Prefill 吞吐（单节点） | 52,300 input tokens/s |
| Decode 吞吐（单节点） | 22,300 output tokens/s |
| 相比 TP 加速比 | 5× |
| 成本 | $0.20/百万 output tokens |
| Inter-Token Latency | ~100ms |

#### 关键技术
1. **RDMA 传输**：KV Cache 通过 RDMA 高速传输，非阻塞后台线程
2. **DisposableTensor 类**：传输完成后立即释放显存，避免 OOM
3. **Two-Batch Overlap**：隐藏通信延迟，在 GPU 计算前启动通信任务
4. **专家并行负载均衡器（EPLB）**：修复 MoE 模型的负载不均问题

**配置示例**：
```bash
# MoE 模型专用参数
--moe-dense-tp-size=1
export SGL_ENABLE_JIT_DEEPGEMM=1
```

#### 生产经验
1. **避免干扰**：PD 分离彻底消除 Prefill 中断 Decode 的问题
2. **延迟权衡**：当前版本优先吞吐量，Inter-Token Latency 为 100ms（可优化）
3. **内存管理**：提前释放传输完成的 tensor，防止显存溢出

---

### 📊 生产案例 3：GB200 NVL72 新硬件部署

**来源**：[LMSYS 博客 - GB200 部署](https://lmsys.org/blog/2025-09-25-gb200-part-2/)

#### 硬件配置
```
系统：GB200 NVL72（NVIDIA Blackwell 架构）
互连带宽：900GB/s 双向链路（远超 H100）
Prefill：2-4 ranks
Decode：48 ranks（大规模 EP）
```

#### 性能提升
| 对比项 | vs H100 加速比 |
|--------|---------------|
| Prefill 阶段 | 3.8× |
| Decode 阶段 | 4.8× |
| 单 GPU Prefill 吞吐 | 26,156 input tokens/s |
| 单 GPU Decode 吞吐 | 13,386 output tokens/s |

#### 关键优化
```bash
# 精度配置
KV Cache: FP8
MoE 专家: NVFP4

# 内核优化
NVIDIA Blackwell DeepGEMM
FlashInfer Blackwell CUTLASS GEMM

# 量化策略
q_b_proj: FP8（替代 BF16）
```

#### 生产经验
1. **精度权衡**：低精度（FP8/NVFP4）几乎无精度损失，通信流量减半
2. **架构特定优化**：Blackwell 高带宽下，细粒度重叠策略更优

---

### 🔧 SGLang PD 分离实战配置

#### A. 单节点快速部署（Llama 模型）

```bash
# 步骤1：启动 Prefill 服务（GPU 0）
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0  # InfiniBand 设备名

# 步骤2：启动 Decode 服务（GPU 1）
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0

# 步骤3：启动路由器
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 \
  --port 8000

# 客户端请求发送到路由器端口 8000
```

#### B. 多节点 DeepSeek 部署（TP + DP + EP）

**Prefill 节点配置**：
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \  # 传输引擎
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  \
  # 分布式配置
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \      # 张量并行
  --dp-size 8 \       # 数据并行
  \
  # MoE 优化
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
```

**Decode 节点配置**：
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  \
  # 分布式配置
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  \
  # Decode 专用优化
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \  # Decode 可处理更多并发
  --cuda-graph-max-bs 64        # CUDA Graph 批次大小
```

#### C. 环境变量调优

**Mooncake NVLink 传输（NVL72 系统）**：
```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

**Prefill 服务器调优**：
| 环境变量 | 作用 | 默认值 |
|---------|------|--------|
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | KV 传输工作线程数 | `int(0.75*CPU核数)//8` (4-12) |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | 并行传输队列数 | 4 |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | 握手超时时间 | 300s |

**Decode 服务器调优**：
| 环境变量 | 作用 | 默认值 |
|---------|------|--------|
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | 健康检查间隔 | 5.0s |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` | 最大失败次数 | 2 |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | KV Cache 接收超时 | 300s |

**NCCL/RDMA 优化（Kubernetes 部署）**：
```bash
# InfiniBand 配置
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_MIN_NCHANNELS=4

# NVShmem 配置
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1

# SGLang 优化
export SGLANG_SET_CPU_AFFINITY=true
export SGLANG_ENABLE_JIT_DEEPGEMM=1
```

### 📝 PD 分离决策树

```
你的模型部署场景？
│
├─ 用户流式输出延迟敏感（聊天机器人）？
│  └─ ✅ 使用 PD 分离（消除 Prefill 中断）
│
├─ 有大量长输入请求（文档问答、代码审查）？
│  └─ ✅ 使用 PD 分离（Prefill 独立扩展）
│
├─ 使用数据并行（DP）且遇到延迟抖动？
│  └─ ✅ 使用 PD 分离（解决 DP 注意力不平衡）
│
├─ 纯批处理场景（离线推理）？
│  └─ ❌ 不需要 PD 分离（统一调度更简单）
│
└─ 成本极度敏感，无法承担额外网络开销？
   └─ ❌ 评估 ROI，5x 吞吐提升可能抵消成本
```

### 📊 性能对比总结

| 部署案例 | GPU 数量 | 架构 | 吞吐提升 | 成本 |
|---------|---------|------|---------|------|
| Kimi K2 | 128 H200 | PD 分离 | 56K/node | $0.21/M tokens |
| DeepSeek | 96 H100 | PD 分离 | 5× vs TP | $0.20/M tokens |
| GB200 | 72 Blackwell | PD 分离 | 4.8× vs H100 | - |
| AMD MI300X | 8 MI300X | PD 分离 | 7× 更好的 SLO | - |

---

## 6️⃣ AI 网关与路由（AI Gateway & Router）

### 🎯 为什么需要 AI 网关？

**问题**：
- 不同请求的复杂度差异巨大（简单问候 vs 复杂推理）
- 使用单一大模型处理所有请求，成本高昂
- 如何在保持质量的同时降低成本？

**解决方案**：智能路由器根据请求复杂度，动态选择最合适的模型

```
                ┌─────────────────┐
     请求 ────►│  AI Gateway     │
                │  (RouteLLM)     │
                └────────┬────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌────────┐     ┌────────┐     ┌────────┐
    │ GPT-4  │     │Mixtral │     │Llama-8B│
    │ (贵)   │     │ (中等) │     │ (便宜) │
    └────────┘     └────────┘     └────────┘
    复杂推理        一般问答        简单对话
```

### 💡 RouteLLM：开源 AI 路由框架

**来源**：[LMSYS 博客 - RouteLLM](https://lmsys.org/blog/2024-07-01-routellm/)

#### 核心原理

RouteLLM 基于 **Chatbot Arena** 的公开对战数据，学习不同模型的优势领域，为每个请求选择最优模型。

**工作流程**：
```
1. 用户请求到达
   ↓
2. 路由器分析请求特征
   ├─ 文本长度
   ├─ 语义复杂度
   ├─ 任务类型（推理/创作/翻译）
   └─ 历史相似请求的模型表现
   ↓
3. 预测模型偏好得分
   ├─ GPT-4: 0.85 (复杂推理)
   ├─ Mixtral: 0.60 (一般问答)
   └─ Llama-8B: 0.30 (简单对话)
   ↓
4. 路由到最优模型
```

#### 四种路由策略

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| **Similarity-Weighted (SW)** | 基于 Elo 评分的相似度加权 | 通用场景 |
| **Matrix Factorization** | 模型-请求兼容性打分 | 多模型混合部署 |
| **BERT 分类器** | 轻量级文本分类 | 低延迟要求 |
| **Causal LLM 分类器** | 使用小模型预测大模型表现 | 高精度要求 |

#### 成本节省效果

| 基准测试 | 成本降低 | 质量保持 |
|---------|---------|---------|
| MT Bench | **85%** | 95% of GPT-4 |
| MMLU | **45%** | 95% of GPT-4 |
| GSM8K | **35%** | 95% of GPT-4 |

**案例**：原本全部使用 GPT-4 的服务，通过 RouteLLM 路由后：
- 30% 的请求路由到 GPT-4（复杂推理）
- 50% 的请求路由到 Mixtral（一般问答）
- 20% 的请求路由到 Llama-8B（简单对话）
- **总成本降低 85%**，质量仅下降 5%

#### 与 SGLang 集成

```bash
# 步骤1：部署多个 SGLang 后端
# GPT-4 后端
python -m sglang.launch_server \
  --model gpt-4 \
  --port 30000

# Mixtral 后端
python -m sglang.launch_server \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --port 30001

# Llama-8B 后端
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 30002

# 步骤2：部署 RouteLLM（独立 Python 服务）
python -m routellm.server \
  --routers similarity_weighted \
  --backends \
    gpt-4=http://localhost:30000 \
    mixtral=http://localhost:30001 \
    llama-8b=http://localhost:30002 \
  --port 8000
```

#### 生产配置建议

**路由器选择矩阵**：
| 场景 | 推荐路由策略 | 原因 |
|------|------------|------|
| 客服聊天 | SW Ranking | 快速响应，适合对话场景 |
| 知识问答 | Matrix Factorization | 准确识别问题难度 |
| 代码生成 | Causal LLM | 高精度，避免误判 |
| 多语言翻译 | BERT Classifier | 轻量级，低延迟 |

**数据增强优化**：
```python
# 使用 LLM Judge 提升路由精度
from routellm import DataAugmentor

augmentor = DataAugmentor(judge_model="gpt-4")
augmentor.augment_training_data(
    original_data="chatbot_arena.jsonl",
    output_path="augmented_data.jsonl"
)
```

### 📊 SGLang 内置路由器（sglang_router）

SGLang 提供原生路由器，支持 PD 分离和多副本负载均衡。

#### 基础用法

```bash
# 多副本负载均衡
python -m sglang_router.launch_router \
  --worker http://localhost:30000 \
  --worker http://localhost:30001 \
  --worker http://localhost:30002 \
  --policy round_robin \  # 或 least_load
  --host 0.0.0.0 \
  --port 8000
```

#### PD 分离路由

```bash
# PD 分离专用路由
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill-service:30000 \
  --decode http://decode-service:30001 \
  --host 0.0.0.0 \
  --port 8000
```

### 🎨 生产架构示例

#### 混合路由 + PD 分离

```
                  ┌─────────────────┐
    外部请求 ────►│   RouteLLM      │
                  │  (智能路由)      │
                  └────────┬────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ GPT-4    │    │ Mixtral  │    │ Llama-8B │
    │ (API)    │    │ PD分离   │    │ 单机     │
    └──────────┘    └────┬─────┘    └──────────┘
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
          ┌────────┐          ┌────────┐
          │Prefill │ RDMA传输 │ Decode │
          │ 4节点  ├─────────►│12节点  │
          └────────┘          └────────┘
```

**配置步骤**：
1. 部署 3 个 SGLang 后端（GPT-4 API/Mixtral PD 分离/Llama-8B 单机）
2. 部署 RouteLLM 路由器
3. 配置路由策略和阈值

---

## 7️⃣ 生产环境最佳实践

`★ Insight ─────────────────────────────────────`
**新增的 PD 分离和 AI 网关章节价值**：
1. **真实案例参考**：Kimi K2（128 H200）、DeepSeek（96 H100）等顶级部署实战经验
2. **成本数据透明**：$0.20-0.21/百万 tokens，RouteLLM 降低 85% 成本
3. **配置即用**：完整的命令行参数和环境变量配置
4. **性能提升显著**：PD 分离带来 5-10x 吞吐提升
`─────────────────────────────────────────────────`

### 🎯 部署清单（Deployment Checklist）

#### A. 监控配置
- [ ] 部署 Prometheus + Grafana
- [ ] 配置告警规则（延迟 > 2s，GPU利用率 > 90%）
- [ ] 设置日志聚合（ELK/Loki）
- [ ] 配置分布式追踪（Jaeger/Zipkin）

#### B. 资源规划
```python
# 显存需求估算公式
GPU_Memory_GB = (
    Model_Size_GB * 1.2  # 模型权重 + 1.2倍安全系数
    + KV_Cache_GB        # 根据 max_total_tokens 计算
    + Activation_GB      # 约为模型大小的 10-20%
)

# 示例：LLaMA-70B FP16
Model_Size = 70B * 2 bytes = 140GB
→ INT4 量化: 140GB / 4 = 35GB
→ 需要 A100 80GB × 1 张（TP=1）
→ 或 A100 40GB × 2 张（TP=2）
```

#### C. 性能调优步骤
1. **Baseline 测试**：记录初始性能指标
2. **瓶颈识别**：使用 Profiler 找出慢在哪里
3. **单点优化**：
   - GPU 利用率低 → 增加批次大小
   - 延迟高 → 启用 TP 或 CUDA Graph
   - 吞吐低 → 启用 DP
4. **A/B 测试**：对比优化前后的 P50/P99 延迟
5. **灰度发布**：先上线 10% 流量验证稳定性

#### D. 故障预案
| 故障类型 | 检测方式 | 恢复措施 | 预防措施 |
|---------|---------|---------|---------|
| OOM 显存溢出 | GPU 显存监控 | 重启服务，减小批次 | 设置 `mem_fraction_static` |
| 请求超时 | P99 延迟告警 | 降级策略（返回缓存） | 设置合理的 `timeout` |
| GPU 卡死 | 健康检查失败 | 自动重启 Worker | 启用 `disable_cuda_graph` |
| 推理卡住 | 心跳超时 | Kill 进程重启 | 设置 `watchdog_timeout` |

### 🎨 生产配置模板

#### 高吞吐在线服务（70B 模型）
```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tp-size 4 \              # 4卡张量并行
  --dp-size 2 \              # 2个副本提高吞吐
  --max-total-tokens 65536 \ # KV缓存大小
  --mem-fraction-static 0.85 \ # GPU 显存利用率
  --schedule-policy sjf \    # 短任务优先
  --enable-metrics \         # 启用 Prometheus 指标
  --disable-radix-cache false \ # 启用 RadixCache
  --port 30000
```

#### 低延迟对话服务（8B 模型）
```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \              # 单卡推理
  --cuda-graph-max-bs 16 \   # 启用 CUDA Graph
  --schedule-conservativeness 0.8 \ # 优先延迟
  --max-total-tokens 32768 \
  --enable-metrics \
  --port 30000
```

#### 成本优化配置（量化 + LoRA）
```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B-Instruct-AWQ \
  --load-format awq \        # AWQ 量化（INT4）
  --tp-size 2 \              # 2卡即可（量化后显存小）
  --enable-lora \            # 启用 LoRA
  --max-loras-per-batch 8 \  # 同时服务8个 LoRA
  --max-total-tokens 49152 \
  --port 30000
```

---

## 8️⃣ 性能基准参考（Benchmarks）

### 🎯 典型硬件配置性能

| 模型 | 硬件 | 配置 | 吞吐量 (TPS) | 延迟 P50 | 延迟 P99 |
|------|------|------|-------------|---------|---------|
| LLaMA-3.1-8B | 1× A100 40GB | TP=1 | 2500 | 45ms | 80ms |
| LLaMA-3.1-70B | 4× A100 80GB | TP=4 | 800 | 120ms | 250ms |
| LLaMA-3.1-405B | 8× A100 80GB | TP=8 | 200 | 300ms | 600ms |
| Qwen2.5-72B | 4× A100 40GB | TP=4, INT4 | 1200 | 100ms | 200ms |

**测试条件**：
- 输入：512 tokens
- 输出：128 tokens
- 并发：32 请求
- 批次大小：自动调整

### 🎯 RadixCache 加速效果

| 场景 | 缓存命中率 | 加速比 | 示例应用 |
|------|-----------|--------|---------|
| 多轮对话 | 60-80% | 3-5× | 客服机器人 |
| Few-shot 提示 | 90%+ | 10×+ | API 服务（固定前缀） |
| 文档问答 | 50-70% | 2-3× | RAG 应用 |
| 代码补全 | 40-60% | 2× | GitHub Copilot 类应用 |

---

## 9️⃣ 故障排查指南（Troubleshooting）

### 🔧 常见问题与解决方案

#### 问题 1：OOM（显存溢出）
**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB
```

**排查步骤**：
1. 检查当前显存占用：`nvidia-smi`
2. 查看配置的 `max_total_tokens`（KV缓存大小）
3. 查看当前批次大小

**解决方案**：
```bash
# 方案1：减小 KV 缓存
--max-total-tokens 32768  # 从 65536 减半

# 方案2：限制批次大小
--max-running-requests 32

# 方案3：减小激活值缓存
--mem-fraction-static 0.75  # 从 0.90 降低

# 方案4：启用量化
--load-format awq  # INT4 量化，显存减少 75%
```

#### 问题 2：延迟突然增加
**症状**：P99 延迟从 200ms 跳到 2000ms

**排查步骤**：
1. 查看 Prometheus 指标：`batch_size_avg`（是否突然变大）
2. 查看 `queue_wait_ms`（队列是否积压）
3. 检查 GPU 利用率（是否达到瓶颈）

**解决方案**：
```bash
# 方案1：限制批次大小（减少 stragglers）
--max-running-requests 16

# 方案2：启用 Prefill 控制
--schedule-conservativeness 0.7

# 方案3：增加 DP 副本
--dp-size 2  # 分流请求

# 方案4：降级策略
# 在业务层实现：超时请求返回缓存或简化回答
```

#### 问题 3：吞吐量低于预期
**症状**：A100 只跑出 500 TPS，预期应该 2000 TPS

**排查步骤**：
1. 查看 GPU 利用率：应该 >80%
2. 查看批次大小：应该 >8
3. 检查是否启用 CUDA Graph
4. 检查是否启用 RadixCache

**解决方案**：
```bash
# 方案1：增加并发请求（填满GPU）
# 客户端侧：增加并发线程数

# 方案2：启用 CUDA Graph
--cuda-graph-max-bs 32

# 方案3：调整调度策略
--schedule-policy sjf  # 减少碎片

# 方案4：减小 TP（单卡足够时）
--tp-size 1  # 避免跨卡通信开销
```

---

## 📚 扩展阅读

### 学术论文
- **Orca**: [Efficient Serving of LLMs with Iteration-Level Scheduling](https://arxiv.org/abs/2209.01188)
- **vLLM**: [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- **FlashInfer**: [Efficient Cascade Inference for LLMs](https://arxiv.org/abs/2401.02984)

### LMSYS 官方博客（生产部署案例）
- **PD 分离系列**：
  - [Kimi K2 部署（128 H200）](https://lmsys.org/blog/2025-07-20-k2-large-scale-ep/)
  - [DeepSeek 部署（96 H100）](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
  - [GB200 NVL72 部署](https://lmsys.org/blog/2025-09-25-gb200-part-2/)
  - [EPD 分离（多模态模型）](https://lmsys.org/blog/2026-01-12-epd-disaggregation/)
- **AI 路由器**：
  - [RouteLLM：成本优化路由框架](https://lmsys.org/blog/2024-07-01-routellm/)
- **生产优化**：
  - [Pipeline Parallelism 百万 Token 上下文](https://lmsys.org/blog/2026-01-15-chunked-pipeline/)
  - [GLM4-MoE 生产优化（65% TTFT 提升）](https://lmsys.org/blog/2026-01-21-novita-glm4/)

### 相关源码

| 功能分类 | 关键目录 |
|---------|---------|
| 监控指标 | `python/sglang/srt/metrics/` |
| 分布式部署 | `python/sglang/srt/managers/` |
| 配置管理 | `python/sglang/srt/configs/` |
| **PD 分离** | **`python/sglang/srt/disaggregation/`** |
| **路由器** | **`python/sglang_router/`** |

---

## 🔗 相关文档

- [← 返回首页](README.md)
- [← 高级功能详解](11-advanced-features.md)
- [→ 术语表](10-glossary.md)
- [→ 请求旅程追踪](04-request-journey.md)
