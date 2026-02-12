# Contribution Guide

Welcome to **SGLang**! We appreciate your interest in contributing. This guide provides a concise overview of how to set up your environment, run tests, build documentation, and open a Pull Request (PR). Whether you're fixing a small bug or developing a major feature, we encourage following these steps for a smooth contribution process.

**中文对照**：欢迎使用 **SGLang**！感谢你的兴趣贡献。本指南概述了如何设置环境、运行测试、构建文档和打开 Pull Request (PR)。无论是修复小错误还是开发主要功能，我们都鼓励遵循这些步骤以获得顺畅的贡献过程。

## Install SGLang from Source

### Fork and clone the repository

**Note**: New contributors do **not** have the write permission to push to the official SGLang repo. Please fork the repository under your GitHub account, then clone your fork locally.

```bash
git clone https://github.com/<your_user_name>/sglang.git
```

**中文对照**：## 从源码安装 SGLang

### Fork 并克隆仓库

**注意**：新贡献者**没有**推送到官方 SGLang 仓库的写入权限。请在你的 GitHub 账户下 fork 仓库，然后在本地克隆你的 fork。

### Build from source

Refer to [Install SGLang from Source](../get_started/install.md#method-2-from-source).

**中文对照**：请参阅[从源码安装 SGLang](../get_started/install.md#method-2-from-source)。

## Format code with pre-commit

We use [pre-commit](https://pre-commit.com/) to maintain consistent code style checks. Before pushing your changes, please run:

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

**中文对照**：## 使用 pre-commit 格式化代码

我们使用 [pre-commit](https://pre-commit.com/) 来保持一致的代码风格检查。在推送更改之前，请运行：

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.
- **Do not commit** directly to the `main` branch. Always create a new branch (e.g., `feature/my-new-feature`), push your changes, and open a PR from that branch.

**中文对照**：
- **`pre-commit run --all-files`** 手动运行所有配置的检查，尽可能应用修复。如果第一次失败，请重新运行以确保 lint 错误完全解决。请确保你的代码在创建 Pull Request 之前通过所有检查。
- **不要直接提交**到 `main` 分支。始终创建一个新分支（例如 `feature/my-new-feature`），推送你的更改，然后从该分支打开 PR。

## Run and add unit tests

If you add a new feature or fix a bug, please add corresponding unit tests to ensure coverage and prevent regression.
SGLang uses Python's built-in [unittest](https://docs.python.org/3/library/unittest.html) framework.
For detailed instructions on running tests and integrating them into CI, refer to [test/README.md](https://github.com/sgl-project/sglang/tree/main/test/README.md).

**中文对照**：## 运行和添加单元测试

如果你添加新功能或修复错误，请添加相应的单元测试以确保覆盖并防止回归。SGLang 使用 Python 内置的 [unittest](https://docs.python.org/3/library/unittest.html) 框架。有关运行测试和将其集成到 CI 的详细说明，请参阅 [test/README.md](https://github.com/sgl-project/sglang/tree/main/test/README.md)。

## Write documentations

We recommend new contributors start from writing documentation, which helps you quickly understand SGLang codebase.
For more details, please refer to [docs/README.md](https://github.com/sgl-project/sglang/tree/main/docs/README.md).

**中文对照**：## 编写文档

我们建议新贡献者从编写文档开始，这有助于你快速了解 SGLang 代码库。更多详情，请参阅 [docs/README.md](https://github.com/sgl-project/sglang/tree/main/docs/README.md)。

## Test the accuracy
If your code changes the model output, please run the accuracy tests. A quick sanity check is the few-shot GSM8K.

```
# Launch a server
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct

# Evaluate
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

**中文对照**：## 测试准确性
如果你的代码更改了模型输出，请运行准确性测试。快速完整性检查是 few-shot GSM8K。

Please note that the above script is primarily a sanity check, not a rigorous accuracy or speed test.
This test can have significant variance (1%–5%) in accuracy due to batching and the non-deterministic nature of the inference engine.
Also, do not rely on the "Latency/Output throughput" from this script, as it is not a proper speed test.

**中文对照**：请注意，上述脚本主要是完整性检查，而不是严格的准确性或速度测试。由于批处理和推理引擎的非确定性性质，此测试的准确性可能会有显著差异（1%–5%）。另外，不要依赖此脚本中的"延迟/输出吞吐量"，因为它不是 proper speed test。

GSM8K is too easy for state-of-the-art models nowadays. Please try your own more challenging accuracy tests.
You can find additional accuracy eval examples in:
- [test_eval_accuracy_large.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_eval_accuracy_large.py)
- [test_gpt_oss_1gpu.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_gpt_oss_1gpu.py)

**中文对照**：对于当今最先进的模型来说，GSM8K 太简单了。请尝试你自己的更具挑战性的准确性测试。你可以在以下位置找到其他准确性评估示例：

## Benchmark the speed
Refer to [Benchmark and Profiling](../developer_guide/benchmark_and_profiling.md).

**中文对照**：## 基准测试速度
请参阅 [Benchmark and Profiling](../developer_guide/benchmark_and_profiling.md)。

## Requesting a review for merge
You can follow the pull request merge process described in [MAINTAINER.md](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md).
You will need to work with the Merge Oncall, Codeowner, and other reviewers to get their approvals.
Then your PR can be merged.

**中文对照**：## 请求审查合并
你可以遵循 [MAINTAINER.md](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md) 中描述的拉取请求合并流程。你需要与 Merge Oncall、Codeowner 和其他审查者合作以获得他们的批准。然后你的 PR 就可以被合并。

## How to Trigger CI Tests

**中文对照**：## 如何触发 CI 测试

We have a lot of open PRs but limited CI machines, so only top and trusted contributors have permission to trigger CI tests.
Users with permission are listed in the [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json)

**中文对照**：我们有很多开放的 PR，但 CI 机器有限，因此只有顶级和受信任的贡献者才有权限触发 CI 测试。有权限的用户列在 [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json) 中。

For CI to run on a pull request, it must have the "run-ci" label. Authorized users can add the label or rerun failed tests by commenting on the PR with one of these commands:

- `/tag-run-ci-label`: Adds the "run-ci" label. Every future commit will trigger CI.
- `/rerun-failed-ci`: Reruns the failed or flaky tests from the most recent commit.
- `/tag-and-rerun-ci`: A single command that performs both `/tag-run-ci-label` and `/rerun-failed-ci`.
- `/rerun-stage <stage-name>`: Reruns a specific test stage without waiting for its dependencies. This is useful when you want to quickly validate a fix for a specific test failure instead of waiting ~30 minutes for preceding stages to complete.

**中文对照**：- `/rerun-stage <stage-name>`：重新运行特定的测试阶段，而无需等待其依赖项完成。当你想快速验证特定测试失败的修复而不是等待约 30 分钟让前面的阶段完成时，这很有用。

If you have permission, the [Slash Command Handler](https://github.com/sgl-project/sglang/actions/workflows/slash-command-handler.yml) will run your command and react with a 👍 to your comment. It may take up to a few minutes for the reaction to appear. Here's a usage [example](https://github.com/sgl-project/sglang/pull/14253#issuecomment-3599509302).

**中文对照**：如果你有权限，[Slash Command Handler](https://github.com/sgl-project/sglang/actions/workflows/slash-command-handler.yml) 将运行你的命令并对你的评论做出 👍 反应。反应可能需要几分钟才能出现。这是一个使用[示例](https://github.com/sgl-project/sglang/pull/14253#issuecomment-3599509302)。

To avoid spamming a PR with too many `/rerun-failed-ci` comments, you can also trigger the command by editing an existing comment and adding any suffix (e.g., `/rerun-failed-ci try again`).

**中文对照**：为避免在 PR 上使用过多 `/rerun-failed-ci` 评论造成垃圾信息，你也可以通过编辑现有评论并添加任何后缀来触发命令（例如 `/rerun-failed-ci try again`）。

Example of rerunning a single test stage: `/rerun-stage unit-test-backend-4-gpu`.

**中文对照**：重新运行单个测试阶段的示例：`/rerun-stage unit-test-backend-4-gpu`。

If you don't have permission, please ask maintainers to trigger CI for you.

**中文对照**：如果你没有权限，请让维护者为你触发 CI。

### CI rate limits

Due to CI scheduling and limited resources, higher-priority PRs may preempt running jobs. In such cases, you may need to rerun the tests.

**中文对照**：### CI 速率限制

由于 CI 调度和有限资源，高优先级 PR 可能会抢占运行中的作业。在这种情况下，你可能需要重新运行测试。

We apply CI rate limits to prevent abuse and ensure fair usage of our CI resources.

**中文对照**：我们应用 CI 速率限制以防止滥用并确保公平使用我们的 CI 资源。

Each CI workflow has a default limit defined in its workflow configuration file. For example, in [pr-gate.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/pr-gate.yml), the default cooldown period is 120 minutes, and each workflow can override it via the `cool-down-minutes` input parameter:

```yaml
cool-down-minutes:
  description: "Default cooldown period in minutes; 0 disables rate limiting"
  type: number
  default: 120
```

**中文对照**：每个 CI 工作流在其工作流配置文件中都有定义的默认限制。例如，在 [pr-gate.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/pr-gate.yml) 中，默认冷却期为 120 分钟，每个工作流可以通过 `cool-down-minutes` 输入参数覆盖它：

Users listed in [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json) may have a per-user cooldown interval. In practice, we use the minimum of the workflow's default window and the user-specific interval.

**中文对照**：列在 [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json) 中的用户可能有每用户冷却间隔。实际上，我们使用工作流默认窗口和用户特定间隔中的最小值。


## Code style guidance

**中文对照**：## 代码风格指南
- 避免代码重复。如果相同的代码片段（超过 5 行）多次出现，将其提取到共享函数中。
- 最小化设备同步。尽可能减少昂贵的 CPU-GPU 同步操作，例如 `tensor.item()` 或 `tensor.cpu()`。使用向量化代码。
- 优先考虑极端效率。SGLang 是一个运行时，你的大部分代码在每个请求的关键路径上运行。尽可能优化所有微小开销，尤其是在模型前向代码中。
  - 一个常见模式是模型前向传递中的一些运行时检查。这些很可能对每一层都相同。请尽可能将结果缓存为单个布尔值。
- 尽可能使函数纯粹。避免对参数进行原地修改。
- 保持文件简洁。如果文件超过 2000 行代码，将其拆分为多个较小的文件。
- 保持测试快速运行。
  - 如果单个测试文件运行时间超过 500 秒，请将其拆分为多个较小的文件。
  - 如果单个 GitHub 工作流作业运行时间超过 30 分钟，请将其拆分为更小的作业/步骤。
  - 在单元测试中重用服务器启动以使测试运行得更快。
- 在支持新硬件或功能时，请遵循以下准则：
  - 不要大幅更改现有代码。
  - 始终优先使用新文件来为新硬件引入特定组件。
  - 如果你为新功能编写多个 if/else 块，请确保常见路径（例如 NVIDIA 硬件或现有代码路径）是第一个分支。
- Avoid code duplication. If the same code snippet (more than five lines) appears multiple times, extract it into a shared function.
- Minimize device synchronization. Reduce expensive CPU-GPU synchronization operations, such as `tensor.item()` or `tensor.cpu()`, whenever possible. Use vectorized code.
- Prioritize extreme efficiency. SGLang is a runtime, and most of your code runs on the critical path for every request. Optimize all minor overheads as much as possible, especially in the model forward code.
  - A common pattern is some runtime checks in the model forward pass (e.g., [this](https://github.com/sgl-project/sglang/blob/f1b0eda55c2c4838e8ab90a0fac7fb1e3d7064ab/python/sglang/srt/models/deepseek_v2.py#L486-L491)). These are very likely the same for every layer. Please cache the result as a single boolean value whenever possible.
- Make functions as pure as possible. Avoid in-place modification of arguments.
- Keep files concise. If a file exceeds 2,000 lines of code, split it into multiple smaller files. (e.g., `scheduler.py`, `scheduler_output_processor_mixin.py`)
- Keep tests run fast.
  - If a single test file run longer than 500 seconds, split it into multiple smaller files (e.g., `test_eagle_infer_a.py`, `test_eagle_infer_b.py`).
  - If a single job in a github workflow runs longer than 30 mins, split it into smaller jobs/steps.
  - Reuse server launches in your unit tests to make tests run faster.
- When supporting new hardware or features, follow these guidelines:
  - Do not drastically change existing code.
  - Always prefer new files to introduce specific components for your new hardware (e.g., `allocator_ascend.py`).
  - If you write multiple if/else blocks for new features, ensure the common path (e.g., NVIDIA hardware or the existing code path) is the first branch.

## How to update sgl-kernel
Since sglang and sgl-kernel are separate Python packages, our current GitHub CI infrastructure does not support updating a kernel and using it immediately within the same pull request (PR).
To add a new kernel or modify an existing one in the sgl-kernel package, you must use multiple PRs.

**中文对照**：## 如何更新 sgl-kernel
由于 sglang 和 sgl-kernel 是独立的 Python 包，我们当前的 GitHub CI 基础设施不支持在同一个拉取请求 (PR) 中更新内核并立即使用它。要在 sgl-kernel 包中添加新内核或修改现有内核，你必须使用多个 PR。

Follow these steps:

**中文对照**：请遵循以下步骤：

1. Submit a PR to update the sgl-kernel source code without using it in sglang python package (e.g., [#8884](https://github.com/sgl-project/sglang/pull/8884/files)).
2. Bump the version of sgl-kernel (e.g., [#9220](https://github.com/sgl-project/sglang/pull/9220/files)).
   - Once merged, this will trigger an automatic release of the sgl-kernel wheel to PyPI.
   - If not urgent, you can wait for other people to release the wheel. A new version will typically be released within one week.
3. Apply the changes:
   - Update the sgl-kernel version in `sglang/python/pyproject.toml` to use the modified kernels.
   - Update the related caller code in the sglang to use the new kernel.

**中文对照**：
1. 提交一个 PR 来更新 sgl-kernel 源代码，而不在 sglang Python 包中使用它（例如 [#8884](https://github.com/sgl-project/sglang/pull/8884/files)）。
2. 提升 sgl-kernel 的版本（例如 [#9220](https://github.com/sgl-project/sglang/pull/9220/files)）。
   - 一旦合并，这将触发 sgl-kernel wheel 自动发布到 PyPI。
   - 如果不紧急，你可以等待其他人发布 wheel。新版本通常会在一周内发布。
3. 应用更改：
   - 更新 `sglang/python/pyproject.toml` 中的 sgl-kernel 版本以使用修改后的内核。
   - 更新 sglang 中的相关调用代码以使用新内核。

## Tips for newcomers

**中文对照**：## 新手提示

If you want to contribute but don't have a specific idea in mind, pick issues labeled ["good first issue" or "help wanted"](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+label%3A%22good+first+issue%22%2C%22help+wanted%22). These tasks typically have lower complexity and provide an excellent introduction to the codebase. Also check out this [code walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through) for a deeper look into SGLang's workflow.

**中文对照**：如果你想贡献但没有具体想法，请选择标记为"good first issue"或"help wanted"的问题。这些任务通常具有较低的复杂度，并且是介绍代码库的绝佳方式。另请查看此代码演练以深入了解 SGLang 的工作流程。

If you have any questions or want to start a discussion, please feel free to ask in our [Slack channel](https://slack.sglang.io).

**中文对照**：如果你有任何问题或想开始讨论，请随时在我们的 Slack 频道中提问。

Thank you for your interest in SGLang. Happy coding!

**中文对照**：感谢你对 SGLang 的兴趣。编码愉快！

## 代码实现

### 贡献者核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/` | 模型架构实现 — 在此添加新模型 |
| `python/sglang/srt/managers/scheduler.py` | 核心调度器 — 修改前请理解连续批处理 |
| `python/sglang/srt/mem_cache/radix_cache.py` | RadixAttention — SGLang 的关键创新，关键路径 |
| `python/sglang/srt/server_args.py` | 所有服务器 CLI 参数 — 在此添加新标志 |
| `test/srt/` | 后端测试目录 — 为新功能添加单元测试 |
| `python/sglang/srt/layers/moe/` | MoE 层实现 — 模块化 EP 框架 |

### 关键开发模式

- **测试框架**：使用 `unittest`（而非 pytest）；通过 `python3 test_file.py` 运行；通过 `setUpClass()` 在测试方法间共享服务器实例
- **CI 注册**：在测试文件中使用 `register_cuda_ci()`；大多数测试选择 `stage-b-test-small-1-gpu`（RTX 5090），FA3/FP8/大型模型选择 `stage-b-test-large-1-gpu`（H100）
- **sgl-kernel 更新**：需要多 PR 流程 — 更新内核源码 → 提升版本 → 更新调用代码
- **代码风格**：无重复（>5 行 → 提取函数）、最小化设备同步、缓存运行时检查、保持文件 <2000 行
