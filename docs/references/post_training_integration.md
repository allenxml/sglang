# Post-Training Integration

SGLang has become the de facto inference backend for modern LLM training frameworks, powering state-of-the-art models across the industry. From GLM-4.6 to Qwen3, leading models leverage SGLang's high-performance inference during reinforcement learning and post-training workflows.

**中文对照**：SGLang 已成为现代 LLM 训练框架的事实推理后端，为业界最先进的模型提供支持。从 GLM-4.6 到 Qwen3，领先模型在强化学习和训练后工作流程中利用 SGLang 的高性能推理。

What makes SGLang essential for post-training?

**中文对照**：是什么让 SGLang 对训练后工作至关重要？

- Open-To-Use Refit Functionality: diverse method for colocate or disaggregate
- Easy To Postpone Generation: enable partial rollout and dedicated rollout control
- Fine-Grained Engine Sleep And Wake Up: facilitate maximum-powered rollout and training
- Training Serving Alignment: ensure the performance consistency in training and serving
- Load Balancing Router: cache-aware load-balancing for high-throughput rollout
- Deterministic Inference: ensure zero kl divergence between rollout and training

**中文对照**：
- 开箱即用的 Refit 功能：多样化的共置或分离方法
- 轻松推迟生成：支持部分展开和专用展开控制
- 细粒度引擎休眠和唤醒：促进最大功率展开和训练
- 训练服务对齐：确保训练和服务中的性能一致性
- 负载均衡路由器：支持缓存感知的高吞吐量展开负载均衡
- 确定性推理：确保展开和训练之间的 KL 散度为零

These capabilities, combined with native integration support across major frameworks, have established SGLang as the infrastructure backbone for modern LLM/VLMs post-training. We also share our latest work in this slide, [Optimizing Large-Scale RL with SGLang](https://gamma.app/docs/Optimizing-RL-with-SGLang-y0kqgj877k34779).

**中文对照**：这些功能与主要框架的原生集成支持相结合，使 SGLang 成为现代 LLM/VLM 训练后的基础设施骨干。我们还在此幻灯片中分享了我们的最新工作，[使用 SGLang 优化大规模 RL](https://gamma.app/docs/Optimizing-RL-with-SGLang-y0kqgj877k34779)。

## Adoption

- [**Miles**](https://github.com/radixark/miles): Enterprise-scale RL framework for large MoE models with SGLang-native rollout, speculative training, and production-grade stability
- [**slime**](https://github.com/THUDM/slime): Post-training framework combining Megatron and SGLang, used to train GLM-4.6
- [**AReaL**](https://github.com/inclusionAI/AReaL): Fully asynchronous RL system achieving 2.77x speedup with SGLang backend for continuous rollout generation
- [**ROLL**](https://github.com/alibaba/ROLL): ROLL is an efficient and user-friendly RL library designed for Large Language Models utilizing Large Scale GPU resources
- [**verl**](https://github.com/volcengine/verl): Full-stack RLHF framework supporting PPO, GRPO, and ReMax with modular SGLang integration
- [**Unsloth**](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide): 2x faster fine-tuning with optimized kernels, deploys seamlessly with SGLang inference
- [**LLaMA Factory**](https://github.com/hiyouga/LLaMA-Factory): Unified framework for training 100+ LLMs with LoRA, QLoRA, and full fine-tuning methods
- [**Tunix**](https://github.com/google/tunix): Google's JAX-native library for LLM post-training with SFT, DPO, PPO, and GRPO support
- [**RL2**](https://github.com/ChenmienTan/RL2): Ray Less Reinforcement Learning, a concise library of post-training for large language models


## Collaboration

Due to the privacy of the design partners, we cannot list the companies that adopt SGLang for post-training. However, we are happy to share the details with you if you are interested and trust the choice among 10+ top companies and frontier labs across US and China. If you are interested in integrating SGLang with your training framework or need technical support, we're here to help! Reach out to us at **rl_team@lmsys.org** for partnerships, integration guidance, and custom feature development.

**中文对照**：由于设计合作伙伴的隐私，我们无法列出采用 SGLang 进行训练后的公司。然而，如果您感兴趣，我们很乐意与美国和中国的 10 多家顶级公司和前沿实验室分享详情。如果您有兴趣将 SGLang 与您的训练框架集成或需要技术支持，我们随时为您提供帮助！请通过 **rl_team@lmsys.org** 联系我们，获取合作伙伴关系、集成指导和定制功能开发支持。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/managers/scheduler.py` | `maybe_sleep_on_idle()` 和 `IdleSleeper` 类：引擎休眠/唤醒机制，用于 RL 训练与推理交替执行 |
| `python/sglang/srt/entrypoints/engine.py` | `Engine` 类：编程式 API，可将 SGLang 直接嵌入训练循环 |
| `python/sglang/srt/server_args.py` | `--sleep-on-idle`、`--enable-deterministic-inference`、`--rl-on-policy-target` 命令行参数 |
| `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py` | `deterministic_all_reduce()`：确保 RL 奖励计算的可复现性 |
| `python/sglang/srt/layers/sampler.py` | 确定性采样路径，保证 rollout 生成的一致性 |

### 集成要点

- **休眠/唤醒循环**：`--sleep-on-idle` 在空闲时释放 GPU 显存，允许训练框架在 rollout 间隙回收资源
- **确定性推理**：`--enable-deterministic-inference` 确保 rollout 与训练之间的 KL 散度为零
- **Engine API**：`sglang.Engine` 提供无需 HTTP 的编程式 generate/encode 接口，可直接嵌入 RL 框架（verl、ROLL 等）
- **权重热更新**：支持不重启服务器的模型权重原地更新（refit），实现 on-policy RL 工作流
