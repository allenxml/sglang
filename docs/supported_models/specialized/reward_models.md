# Reward Models

These models output a scalar reward score or classification result, often used in reinforcement learning or content moderation tasks.

**中文对照**：奖励模型

这些模型输出标量奖励分数或分类结果，常用于强化学习或内容审核任务。

```{important}
They are executed with `--is-embedding` and some may require `--trust-remote-code`.
```

## Example launch Command

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Math-RM-72B \  # example HF/local path
  --is-embedding \
  --host 0.0.0.0 \
  --tp-size=4 \                          # set for tensor parallelism
  --port 30000 \
```

**中文对照**：示例启动命令

## Supported models

**中文对照**：支持的模型

| Model Family (Reward)                                                     | Example HuggingFace Identifier                              | Description                                                                     |
|---------------------------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------|
| **Llama (3.1 Reward / `LlamaForSequenceClassification`)**                   | `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`            | Reward model (preference classifier) based on Llama 3.1 (8B) for scoring and ranking responses for RLHF.  |
| **Gemma 2 (27B Reward / `Gemma2ForSequenceClassification`)**                | `Skywork/Skywork-Reward-Gemma-2-27B-v0.2`             | Derived from Gemma‑2 (27B), this model provides human preference scoring for RLHF and multilingual tasks.  |
| **InternLM 2 (Reward / `InternLM2ForRewardMode`)**                         | `internlm/internlm2-7b-reward`                       | InternLM 2 (7B)–based reward model used in alignment pipelines to guide outputs toward preferred behavior.  |
| **Qwen2.5 (Reward - Math / `Qwen2ForRewardModel`)**                         | `Qwen/Qwen2.5-Math-RM-72B`                           | A 72B math-specialized RLHF reward model from the Qwen2.5 series, tuned for evaluating and refining responses.  |
| **Qwen2.5 (Reward - Sequence / `Qwen2ForSequenceClassification`)**          | `jason9693/Qwen2.5-1.5B-apeach`                      | A smaller Qwen2.5 variant used for sequence classification, offering an alternative RLHF scoring mechanism.  |

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/entrypoints/http_server.py` | `/classify` API 端点：处理奖励模型的评分请求 |
| `python/sglang/srt/models/` | 奖励模型实现：如 `LlamaForSequenceClassification`、`Qwen2ForRewardModel`、`InternLM2ForRewardModel` |
| `python/sglang/srt/server_args.py` | `--is-embedding` 标志：奖励模型需要此标志启用嵌入运行模式 |

### 集成要点

- **启动方式**：所有奖励模型需要 `--is-embedding` 标志，部分需要 `--trust-remote-code`
- **输出格式**：奖励模型输出标量分数或分类结果，常用于 RLHF 对齐训练和内容审核
- **TP 支持**：大规模奖励模型（如 Qwen2.5-Math-RM-72B）可通过 `--tp-size` 进行张量并行
- **与分类 API 的关系**：`/classify` 端点最初为奖励模型设计，现已扩展支持所有非生成式模型
