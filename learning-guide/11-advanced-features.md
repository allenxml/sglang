# 🚀 高级功能详解

> **适合读者**：已掌握 SGLang 基础推理流程，想深入了解结构化输出、函数调用、多模态和 LoRA 等高级特性。

---

## 📚 本章内容概览

| 功能模块 | 典型应用场景 | 核心价值 |
|---------|-------------|---------|
| 结构化输出 | API响应、数据提取 | 强制模型输出符合格式（JSON/正则/EBNF） |
| 函数调用 | AI Agent、工具集成 | 让模型能调用外部工具（搜索、计算器等） |
| 多模态 | 图文理解、OCR | 视觉-语言模型的图像处理 |
| LoRA | 多租户服务 | 一个基座模型服务多个定制版本 |

---

## 1️⃣ 结构化输出（Constrained Decoding）

### 🎯 解决什么问题？

**问题**：大模型输出不可控，可能生成格式错误的 JSON 或不符合要求的文本。

**例子**：
```python
# 期望输出（合法JSON）
{"name": "张三", "age": 25, "city": "北京"}

# 实际输出（模型可能犯的错误）
{name: 张三, age: 25, city: 北京}  # 缺引号
{"name": "张三", "age": "25岁"}     # age应该是数字
这是张三的信息：姓名张三，年龄25...  # 完全不是JSON
```

### 💡 解决方案：XGrammar 约束解码

**核心思想**：用有限状态机（FSM）在生成**每个token前**检查是否符合语法规则。

**比喻理解**：像"交通灯系统"
- 每个状态 = 一个路口
- 红灯（mask=0）= 这个token违反语法，禁止通过
- 绿灯（mask=1）= 这个token合法，允许通过

### 🔧 技术实现流程

```
用户请求 + JSON Schema
    ↓
[GrammarManager] 语法管理器
    ├─ 检查缓存（避免重复编译）
    ├─ 异步编译：JSON Schema → FSM
    └─ 分布式同步（DP/TP环境）
    ↓
[XGrammarBackend] FSM后端
    ├─ 初始状态：{}
    ├─ 每一步：计算合法token集合
    │   例如：刚输入{，下一个只能是"或}
    └─ 状态转移：根据已生成token更新FSM
    ↓
[Sampler] 采样器应用掩码
    ├─ 合法token：logit保持不变
    └─ 非法token：logit = -∞（概率=0）
    ↓
生成结果：100%符合JSON Schema ✅
```

### 📝 关键代码文件

| 文件 | 核心职责 |
|------|---------|
| `grammar_manager.py` | 编译缓存、异步处理、跨进程同步 |
| `xgrammar_backend.py` | FSM构建、token掩码计算、状态转移 |

**核心方法**：
- `GrammarManager.process_req_with_grammar()` - 检测请求是否需要语法约束
- `XGrammarGrammar.fill_vocab_mask()` - 填充当前状态的合法token掩码
- `XGrammarGrammar.accept_token()` - 接受token并转移FSM状态

### 📊 性能特点

- **编译开销**：首次编译语法 ~100ms，后续命中缓存 ~1ms
- **推理开销**：每步约增加 5-10% 延迟（相比无约束生成）
- **准确率**：100%（数学保证生成符合语法）

### 🎨 支持的语法类型

| 类型 | 示例 | 应用场景 |
|------|------|---------|
| JSON Schema | `{"type": "object", "properties": {...}}` | API响应、结构化数据 |
| 正则表达式 | `\d{3}-\d{4}-\d{4}` | 电话号码、邮箱验证 |
| EBNF语法 | `root ::= "hello" (" " ("world"\|"there"))` | 自定义DSL |
| Structural Tag | `<tool_call>{...}</tool_call>` | 函数调用格式 |

---

## 2️⃣ 函数调用（Function Calling）

### 🎯 解决什么问题？

让大模型能调用外部工具，成为真正的 AI Agent。

**例子**：
```
用户：北京明天天气怎么样？

[无函数调用]
模型：我无法实时获取天气信息... (幻觉回答)

[有函数调用]
模型：<tool_call>{"name": "get_weather", "args": {"city": "北京", "date": "明天"}}</tool_call>
系统：调用API → 返回"晴天，15-25℃"
模型：根据查询结果，北京明天晴天，气温15-25℃
```

### 💡 解决方案：多模型格式解析

**问题**：不同模型的函数调用格式不同
- Qwen: `<tool_call>...</tool_call>`
- Llama: `<function>...</function>`
- DeepSeek: 自定义JSON格式

**SGLang方案**：统一解析框架 + 模型特定检测器

### 🔧 技术实现流程

```
模型生成文本（包含工具调用）
    ↓
[FunctionCallParser] 统一解析入口
    ├─ 根据模型类型选择Detector
    │   Qwen → Qwen25Detector
    │   Llama → Llama32Detector
    │   DeepSeek → DeepSeekV3Detector
    └─ 支持流式和非流式解析
    ↓
[BaseFormatDetector] 格式检测基类
    ├─ has_tool_call(): 快速判断是否包含调用
    ├─ parse_non_stream(): 一次性解析完整文本
    └─ parse_streaming_increment(): 增量解析流式chunk
    ↓
提取结果：List[ToolCallItem]
    ├─ tool_index: 工具编号
    ├─ name: 函数名（如 get_weather）
    └─ parameters: JSON参数（如 {"city": "北京"}）
```

### 📝 关键代码文件

| 文件 | 核心职责 |
|------|---------|
| `function_call_parser.py` | 统一解析入口、模型注册表、约束生成 |
| `base_format_detector.py` | 抽象基类、流式状态管理、JSON解析 |

**核心方法**：
- `FunctionCallParser.parse_stream_chunk()` - 流式增量解析
- `BaseFormatDetector.parse_streaming_increment()` - 跨chunk状态维护
- `FunctionCallParser.get_structure_constraint()` - 生成函数调用的语法约束

### 🎨 流式解析挑战

**问题**：工具调用的JSON可能跨多个流式chunk
```python
# Chunk 1: '{"name": "get_wea'
# Chunk 2: 'ther", "args": {"ci'
# Chunk 3: 'ty": "北京"}}'
```

**解决方案**：
1. 维护内部缓冲区累积不完整的JSON
2. 使用 `partial_json_parser` 解析未闭合的JSON
3. 检测完整性后才发送到下游
4. 跟踪已发送内容，计算增量差异

### 📊 支持的模型

| 模型系列 | Detector类 | 格式特点 |
|---------|-----------|---------|
| Qwen 2.5 | Qwen25Detector | `<tool_call>[{...}]</tool_call>` |
| Llama 3.2 | Llama32Detector | `<function>...</function>` |
| DeepSeek V3 | DeepSeekV3Detector | 自定义JSON数组 |
| Mistral | MistralDetector | `[TOOL_CALLS] [...]` |
| GLM-4 | Glm4MoeDetector | 特殊XML标记 |

---

## 3️⃣ 多模态（Multi-Modal）

### 🎯 解决什么问题？

处理图像+文本的混合输入，让模型"看懂"图片。

**典型应用**：
- 📷 图文理解："这张图片里有什么？"
- 📄 OCR提取："提取这张发票的金额"
- 🎨 视觉问答："图中的建筑是什么风格？"

### 💡 解决方案：Vision Encoder + Language Model

**架构**：
```
图像输入 (PIL Image)
    ↓
[mm_utils] 图像预处理
    ├─ load_image_from_base64(): Base64解码
    ├─ select_best_resolution(): 自适应分辨率
    ├─ split_to_patches(): 切分为patch
    └─ 归一化 + 张量转换
    ↓
[ViT] 视觉编码器 (Vision Transformer)
    ├─ 提取视觉特征
    ├─ 输出：视觉token序列
    └─ [ViTCudaGraphRunner] CUDA Graph加速
    ↓
[Projection] 投影层
    ├─ 将视觉特征映射到语言空间
    └─ 输出维度 = LLM的嵌入维度
    ↓
[Language Model] 语言模型
    ├─ 输入：[图像token] + [文本token]
    └─ 联合理解，生成回答
```

### 🔧 关键技术：Anyres 处理

**问题**：不同图像分辨率差异大（从 224×224 到 4096×4096）

**Anyres方案**：动态选择最佳分辨率
```python
原始图像: 1920×1080
    ↓
候选分辨率: [(672,384), (1344,768), (1008,576), ...]
    ↓
select_best_resolution()
    计算每个候选的缩放比例
    选择最接近原始比例的分辨率
    ↓
选中: 1344×768 (比例最接近16:9)
    ↓
分块处理
    全局视图: 384×384 (缩略图)
    局部patch: 7个 336×336 的patch
    总token数: 1 + 7 = 8个视觉块
```

### 🚀 性能优化：CUDA Graph

**问题**：ViT的多层Transformer计算涉及大量小kernel，CPU-GPU通信开销高。

**ViTCudaGraphRunner 解决方案**：
```python
# 第一次运行（慢，但记录操作）
第一帧图像 → 执行ViT → 记录GPU操作序列 → 保存为Graph

# 后续运行（快，直接回放）
相同尺寸图像 → 回放Graph → 跳过CPU调度开销

性能提升: 20-40% 延迟降低 ⚡
```

### 📝 关键代码文件

| 文件 | 核心职责 |
|------|---------|
| `mm_utils.py` | 图像加载、Anyres处理、分块、格式转换 |
| `vit_cuda_graph_runner.py` | ViT的CUDA Graph捕获与回放 |

**核心方法**：
- `load_image_from_base64()` - Base64字符串 → PIL图像
- `select_best_resolution()` - 自适应分辨率选择
- `split_to_patches()` - 大图切分为多个patch
- `ViTCudaGraphRunner.run()` - CUDA Graph加速执行

### 📊 支持的多模态模型

| 模型 | 视觉编码器 | 特殊处理 |
|------|-----------|---------|
| LLaVA-NeXT | CLIP ViT | Anyres分块 |
| Qwen2-VL | ViT-G | 窗口注意力 |
| Qwen3-VL | ViT-G | DeepStack merger |
| InternVL | InternViT | 动态分辨率 |

---

## 4️⃣ LoRA 动态适配（Low-Rank Adaptation）

### 🎯 解决什么问题？

**场景**：多租户需要同一模型的不同定制版本
- 客户A：金融领域专用（风控术语、合规话术）
- 客户B：医疗领域专用（病历书写、药物咨询）
- 客户C：客服领域专用（话术模板、情绪识别）

**传统方案的问题**：
```
方案1: 为每个客户部署独立模型
→ GPU显存爆炸（3个70B模型 = 420GB显存）

方案2: 为每个客户微调完整模型
→ 训练成本高，存储开销大（每个模型70GB）
```

### 💡 解决方案：LoRA 低秩适配

**核心思想**：冻结基座模型，只训练小矩阵

**数学原理**：
```
传统微调: 更新整个权重矩阵 W (4096×4096)
→ 参数量: 16M

LoRA: W 保持不变，添加 A (4096×8) 和 B (8×4096)
→ 参数量: 8×4096×2 = 65K (减少 250倍！)

前向传播: y = Wx + s·BAx
         ↑      ↑
      基座输出  LoRA增量（缩放因子s控制强度）
```

**比喻**："给西装加补丁"
- 基座模型W = 原装西装（大且不可改）
- LoRA矩阵A、B = 小补丁（轻便可更换）
- 最终效果 = 原装 + 补丁

### 🔧 多LoRA并发服务

**SGLang 的 S-LoRA 实现**：
```
单个GPU (A100 80GB)
    ↓
[基座模型] LLaMA-70B (140GB FP16 → 70GB INT4量化)
    ↓
剩余显存 10GB 可容纳 100+ 个LoRA适配器（每个~100MB）
    ↓
[LoRAManager] 动态调度
    ├─ 请求1 → 加载LoRA_金融
    ├─ 请求2 → 加载LoRA_医疗
    ├─ 请求3 → 加载LoRA_客服
    └─ 请求4 → 复用LoRA_金融（已加载）
    ↓
[LoRA层] 高效矩阵运算
    ├─ 批次合并：同一LoRA的请求合并计算
    ├─ Punica kernel：优化的CUDA kernel
    └─ 动态权重加载：按需从CPU/磁盘加载LoRA权重
```

### 📝 关键代码文件

| 文件 | 核心职责 |
|------|---------|
| `lora_manager.py` | LoRA生命周期管理、请求路由、内存池 |
| `lora/layers.py` | LoRA数学实现（y = Wx + s·BAx） |

**核心方法**：
- `LoRAManager.add_lora()` - 加载新的LoRA适配器
- `LoRAManager.remove_lora()` - 卸载不常用的LoRA
- `BaseLayerWithLoRA.forward()` - 基座输出 + LoRA增量

### 🎨 LoRA 层类型

| 层类型 | LoRA实现 | 技术细节 |
|--------|---------|---------|
| Linear | `ColumnParallelLinearWithLoRA` | 列并行切分 |
| Linear | `RowParallelLinearWithLoRA` | 行并行切分 + all-reduce |
| QKV | `QKVParallelLinearWithLoRA` | Q、K、V独立LoRA |
| Embedding | `VocabParallelEmbeddingWithLoRA` | 词嵌入层LoRA |

### 📊 性能与成本

**内存效率**：
```
基座模型: 70GB
100个LoRA: 100 × 100MB = 10GB
总显存: 80GB (单卡A100可容纳)

传统方案: 70GB × 100 = 7TB（需要100张A100！）
节省: 98.9% GPU成本
```

**推理延迟**：
```
基座推理: 100ms
+LoRA计算: 5-10ms (额外5-10%开销)
总延迟: 105-110ms

收益: 用10%性能开销换取100倍并发能力
```

### 🔥 适用场景

| 场景 | 说明 | 效果 |
|------|------|------|
| 多租户SaaS | 每个企业客户一个LoRA | 成本降低98%，按需计费 |
| A/B测试 | 同时运行多个微调版本 | 快速迭代，实时对比 |
| 领域适配 | 医疗、金融、法律等垂直领域 | 专业能力提升，通用性保持 |

---

## 🎓 学习建议

### 优先级排序（从高到低）

1. **必学**（生产必备）：
   - ✅ 结构化输出 - API开发必需
   - ✅ 函数调用 - Agent应用核心

2. **重要**（高级场景）：
   - ⭐ LoRA - 多租户/定制化服务
   - ⭐ 多模态 - VLM应用

3. **补充**（性能优化）：
   - CUDA Graph优化细节
   - FSM编译缓存策略

### 实践路径

```
第1周：结构化输出
├─ 阅读 grammar_manager.py 注释
├─ 实验：使用JSON Schema约束API响应
└─ 深入：理解FSM状态转移机制

第2周：函数调用
├─ 阅读 function_call_parser.py 注释
├─ 实验：实现天气查询Agent
└─ 深入：流式解析的状态机维护

第3周：多模态（可选）
├─ 阅读 mm_utils.py 注释
├─ 实验：图文理解应用
└─ 深入：Anyres和CUDA Graph优化

第4周：LoRA（可选）
├─ 阅读 lora_manager.py 注释
├─ 实验：部署多个定制LoRA
└─ 深入：S-LoRA调度算法
```

---

## 📚 扩展阅读

### 学术论文

- **XGrammar**: [Efficient Structured Generation for Large Language Models](https://arxiv.org/abs/2410.19392)
- **S-LoRA**: [Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285)
- **Punica**: [Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547)
- **LLaVA-NeXT**: [Improved Reasoning, OCR, and World Knowledge](https://arxiv.org/abs/2404.15843)

### 相关源码

| 功能 | 关键目录 |
|------|---------|
| 约束解码 | `python/sglang/srt/constrained/` |
| 函数调用 | `python/sglang/srt/function_call/` |
| 多模态 | `python/sglang/srt/multimodal/` |
| LoRA | `python/sglang/srt/lora/` |

---

## 🔗 相关文档

- [← 返回首页](README.md)
- [← 采样与生成](09-sampling.md)
- [→ 生产环境部署](12-production-deployment.md)
- [→ 术语表](10-glossary.md)
