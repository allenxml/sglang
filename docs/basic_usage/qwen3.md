# Qwen3-Next Usage

SGLang has supported Qwen3-Next-80B-A3B-Instruct and Qwen3-Next-80B-A3B-Thinking since [this PR](https://github.com/sgl-project/sglang/pull/10233).

**中文对照**：自[此 PR](https://github.com/sgl-project/sglang/pull/10233) 起，SGLang 已支持 Qwen3-Next-80B-A3B-Instruct 和 Qwen3-Next-80B-A3B-Thinking。

## Launch Qwen3-Next with SGLang

**中文对照**：使用 SGLang 启动 Qwen3-Next

To serve Qwen3-Next models on 4xH100/H200 GPUs:

**中文对照**：要在 4xH100/H200 GPU 上服务 Qwen3-Next 模型：

```bash
python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4
```

### Configuration Tips

**中文对照**：配置提示

- `--max-mamba-cache-size`: Adjust `--max-mamba-cache-size` to increase mamba cache space and max running requests capability. It will decrease KV cache space as a trade-off. You can adjust it according to workload.

**中文对照**：- `--max-mamba-cache-size`：调整 `--max-mamba-cache-size` 以增加 mamba 缓存空间和最大运行请求能力。作为权衡，它会减少 KV 缓存空间。你可以根据工作负载进行调整。

- `--mamba-ssm-dtype`: `bfloat16` or `float32`, use `bfloat16` to save mamba cache size and `float32` to get more accurate results. The default setting is `float32`.

**中文对照**：- `--mamba-ssm-dtype`：`bfloat16` 或 `float32`，使用 `bfloat16` 可以节省 mamba 缓存大小，`float32` 可以获得更准确的结果。默认设置为 `float32`。

- `--mamba-full-memory-ratio`: The ratio of mamba state memory to full kv cache memory. The default is 0.9.

**中文对照**：- `--mamba-full-memory-ratio`：mamba 状态内存与完整 kv 缓存内存的比率。默认值为 0.9。

### Mamba Radix Cache

**中文对照**：Mamba Radix 缓存

SGLang supports prefix caching for Qwen3-Next models named `MambaRadixCache`, which improves inference speed by reusing computation results. There are two versions of `MambaRadixCache`:

**中文对照**：SGLang 支持名为 `MambaRadixCache` 的 Qwen3-Next 模型的前缀缓存，通过重用计算结果来提高推理速度。`MambaRadixCache` 有两个版本：

- `no_buffer`: The default version, which is also other hybrid linear models' choice. When it is enabled, SGLang will automatically close overlap schedule for compatibility reasons.

**中文对照**：- `no_buffer`：默认版本，也是其他混合线性模型的选择。启用时，SGLang 将自动关闭重叠调度以保持兼容性。

- `extra_buffer`: An optimized version that is compatible with features like page size > 1, overlap schedule, and speculative decoding. It also supports storing mamba state in branching positions. However, it requires two extra mamba spaces for a ping-pong buffer for each request. To enable it, add the argument `--mamba-scheduler-strategy extra_buffer` when launching the server.

**中文对照**：- `extra_buffer`：一个优化版本，兼容页大小 > 1、重叠调度和推测解码等功能。它还支持在分支位置存储 mamba 状态。但是，它需要为每个请求提供两个额外的 mamba 空间用于乒乓缓冲区。要启用它，在启动服务器时添加参数 `--mamba-scheduler-strategy extra_buffer`。

### EAGLE Speculative Decoding

**中文对照**：EAGLE 推测解码

**Description**: SGLang has supported Qwen3-Next models with [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding).

**中文对照**：**描述**：SGLang 通过 [EAGLE 推测解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding) 支持 Qwen3-Next 模型。

**Usage**:

**中文对照**：**用法**：

Add arguments `--speculative-algorithm`, `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` to enable this feature. For example:

**中文对照**：添加参数 `--speculative-algorithm`、`--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 以启用此功能。例如：

``` bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp 4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-algo NEXTN
```

Details can be seen in [this PR](https://github.com/sgl-project/sglang/pull/10233).

**中文对照**：详情可见于[此 PR](https://github.com/sgl-project/sglang/pull/10233)。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/qwen3_moe.py` | Qwen3-Next 模型架构，包含 Mamba + Transformer 混合层 |
| `python/sglang/srt/mem_cache/radix_cache.py` | `MambaRadixCache`：混合线性模型的前缀缓存（no_buffer / extra_buffer 模式） |
| `python/sglang/srt/speculative/eagle_worker.py` | Qwen3-Next 的 NEXTN 推测解码算法 |

### 集成要点

- **服务器参数**：`--max-mamba-cache-size`、`--mamba-ssm-dtype bfloat16|float32`、`--mamba-full-memory-ratio`、`--mamba-scheduler-strategy no_buffer|extra_buffer`
- **推测解码**：`--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
