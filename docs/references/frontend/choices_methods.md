# Choices Methods in SGLang
This doc describes the choices methods supported by SGLang.

The optional `choices_method` arg determines how options supplied to SGLang's `choices` primitive are selected. Only the `RuntimeEndpoint` backend supports the `choices_method` arg. Other backends, such as `OpenAI`, have bespoke selection implementations due to API limitations.

## Methods

### Token Length Normalized

Token length normalized is the default SGLang choices method. It selects the option with the highest average logprob across all of its tokens.

Usage example (alternatively, simply omit the `choices_method` arg):
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.token_length_normalized,
        )
    )
```


This can perform poorly if an option contains many tokens, where its later tokens are predicted with high confidence based on its earlier tokens. For instance, even strong models will fail the above example if the specified options are `["Paris", "Antidisestablishmentarianism"]`.

### Greedy Token Selection

Greedy token selection simply selects the option with the highest logprob for its initial token. For overlapping options where one option is a subset of a longer option, the logprobs of the shorter option are extended using its average logprob for comparison against the longer option.

Usage example:
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.greedy_token_selection,
        )
    )
```

This can perform poorly if an option misleads the model down a bad path based on an attractive initial token. For instance, greedy selection will result in an incorrect response for this example:
```python
@sgl.function
def us_president_example(s):
    s += sgl.user("Name a US president.")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["Donald Duck", "Millard Fillmore"],
            choices_method=sgl.greedy_token_selection,
        )
    )
```

### Unconditional Likelihood Normalized

Unconditional likelihood normalized selects the option with the highest average token logprob once normalized by the unconditional token logprobs, as described in [this EleutherAI blogpost](https://blog.eleuther.ai/multiple-choice-normalization/). This method incurs an additional LLM call to obtain the unconditional likelihoods.

Usage example:
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.unconditional_likelihood_normalized,
        )
    )
```

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/lang/choices.py` | 选择方法实现：`TokenLengthNormalized`、`GreedyTokenSelection`、`UnconditionalLikelihoodNormalized` 类 |
| `python/sglang/__init__.py` | 导出 `sgl.token_length_normalized`、`sgl.greedy_token_selection`、`sgl.unconditional_likelihood_normalized` |

### 关键代码逻辑

- **Token 长度归一化**（默认方法）：计算所有 token 的平均 logprob，选择平均值最高的选项
- **贪心 token 选择**：仅使用首个 token 的 logprob；对较短选项用其平均 logprob 延伸以进行长度对齐比较
- **无条件似然归一化**：从条件 logprob 中减去无条件（无上下文）logprob 后再取平均；需要额外一次 LLM 调用

### 集成要点

- **后端要求**：仅 `RuntimeEndpoint`（SGLang 原生后端）支持 `choices_method`；OpenAI 后端使用自有实现
- **使用方式**：在 `@sgl.function` 装饰的函数中使用 `sgl.gen("answer", choices=[...], choices_method=sgl.token_length_normalized)`
- **依赖 logprob**：所有方法都依赖服务器为每个候选选项返回逐 token 的 logprob
