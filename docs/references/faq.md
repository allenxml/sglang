# Troubleshooting and Frequently Asked Questions

## Troubleshooting

This page lists common errors and tips for resolving them.

**中文对照**：本页列出了常见错误和解决技巧。

### CUDA Out of Memory
If you encounter out-of-memory (OOM) errors, you can adjust the following parameters:

- If OOM occurs during prefill, try reducing `--chunked-prefill-size` to `4096` or `2048`. This saves memory but slows down the prefill speed for long prompts.
- If OOM occurs during decoding, try lowering `--max-running-requests`.
- You can also decrease `--mem-fraction-static` to a smaller value, such as 0.8 or 0.7. This decreases the memory usage of the KV cache memory pool and helps prevent OOM errors during both prefill and decoding. However, it limits maximum concurrency and reduces peak throughput.
- Another common case for OOM is requesting input logprobs for a long prompt as it requires significant memory. To address this, set `logprob_start_len` in your sampling parameters to include only the necessary parts. If you do need input logprobs for a long prompt, try reducing `--mem-fraction-static`.

### CUDA Error: Illegal Memory Access Encountered
This error may result from kernel errors or out-of-memory issues:
- If it is a kernel error, resolving it may be challenging. Please file an issue on GitHub.
- If it is an out-of-memory issue, it may sometimes be reported as this error instead of "Out of Memory." Refer to the section above for guidance on avoiding OOM issues.

**中文对照**：此错误可能由内核错误或内存不足问题导致：
- 如果是内核错误，解决起来可能很有挑战性。请在 GitHub 上提交 issue。
- 如果是内存不足问题，它有时会报告为此错误而不是"Out of Memory"。请参阅上文关于避免 OOM 问题的指导。

### The server hangs
- If the server hangs during initialization or running, it can be memory issues (out of memory), network issues (nccl errors), or other bugs in sglang.
    - If it is out of memory, you might see that `avail mem` is very low during the initialization or right after initialization. In this case,
      you can try to decrease `--mem-fraction-static`, decrease `--cuda-graph-max-bs`, or decrease `--chunked-prefill-size`.
- Other bugs, please file an issue on GitHub.


## Frequently Asked Questions

### The results are not deterministic, even with a temperature of 0

You may notice that when you send the same request twice, the results from the engine will be slightly different, even when the temperature is set to 0.

**中文对照**：您可能会注意到，即使温度设置为 0，当您发送相同的请求两次时，引擎的结果也会略有不同。

From our initial investigation, this indeterminism arises from two factors: dynamic batching and prefix caching. Roughly speaking, dynamic batching accounts for about 95% of the indeterminism, while prefix caching accounts for the remaining portion. The server runs dynamic batching under the hood. Different batch sizes can cause PyTorch/CuBLAS to dispatch to different CUDA kernels, which can lead to slight numerical differences. This difference accumulates across many layers, resulting in nondeterministic output when the batch size changes. Similarly, when prefix caching is enabled, it can also dispatch to different kernels. Even when the computations are mathematically equivalent, small numerical differences from different kernel implementations lead to the final nondeterministic outputs.

**中文对照**：根据我们的初步调查，这种不确定性源于两个因素：动态批处理和前缀缓存。粗略地说，动态批处理约占不确定性的 95%，而前缀缓存占剩下的部分。服务器在后台运行动态批处理。不同的批大小可能导致 PyTorch/CuBLAS 分派到不同的 CUDA 内核，这可能导致轻微的数值差异。这种差异在许多层中累积，导致批大小变化时产生非确定性输出。类似地，当启用前缀缓存时，它也可能分派到不同的内核。即使计算在数学上等效，不同内核实现的小数值差异也会导致最终的确定性输出。

To achieve more deterministic outputs in the current code, you can add `--disable-radix-cache` and send only one request at a time. The results will be mostly deterministic under this setting.

**中文对照**：要在当前代码中获得更确定性的输出，您可以添加 `--disable-radix-cache` 并每次只发送一个请求。在此设置下，结果将基本确定。

**Update**:
Recently, we also introduced a deterministic mode, you can enable it with `--enable-deterministic-inference`.
Please find more details in this blog post: https://lmsys.org/blog/2025-09-22-sglang-deterministic/

**中文对照**：
**更新**：
最近，我们还引入了确定性模式，您可以使用 `--enable-deterministic-inference` 启用它。
请在此博客文章中查找更多详情：https://lmsys.org/blog/2025-09-22-sglang-deterministic/

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/server_args.py` | `--mem-fraction-static`、`--chunked-prefill-size`、`--max-running-requests`、`--enable-deterministic-inference` 命令行参数 |
| `python/sglang/srt/mem_cache/memory_pool.py` | GPU 显存池：`mem_fraction_static` 控制 KV 缓存的显存分配比例 |
| `python/sglang/srt/managers/scheduler.py` | 连续批处理调度器：`max_running_requests` 限制并发解码数量；动态批处理是非确定性的主要来源 |
| `python/sglang/srt/mem_cache/radix_cache.py` | RadixCache：前缀缓存可能分派不同内核，是非确定性的次要来源 |
| `python/sglang/srt/layers/sampler.py` | 确定性采样路径，由 `--enable-deterministic-inference` 激活 |
| `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py` | `deterministic_all_reduce()`：确保集合通信操作的可复现性 |

### 集成要点

- **OOM 预防**：`--mem-fraction-static 0.7` 减少 KV 缓存池大小；`--chunked-prefill-size 2048` 限制单次 prefill 的显存占用
- **确定性模式**：`--enable-deterministic-inference` 自动切换注意力和采样后端为确定性实现
- **非确定性来源**：动态批处理（约 95%）和前缀缓存（约 5%）导致不同 CUDA 内核被调度 → 数值差异逐层累积
- **临时方案**：`--disable-radix-cache` + 每次只发送一个请求可获得基本确定的结果，无需启用完整确定性模式
