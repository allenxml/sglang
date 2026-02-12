# Cuda Graph for Multi-Modal Encoder in SGLang

## Motivation

In multimodal reasoning services, the visual encoder (ViT / Vision Transformer) typically has a few characteristic traits:

Many layers, fragmented operators: Each layer includes LN, QKV projections, attention, MLP, residual connections, etc., resulting in extremely frequent kernel launches.

**中文对照**：## 动机

在多模态推理服务中，视觉编码器（ViT / Vision Transformer）通常具有以下几个特征：

层数众多，运算符碎片化：每层包含 LN、QKV 投影、注意力、MLP、残差连接等，导致内核启动非常频繁。

Server-side "small batch / low latency" is common: The batch size is very small (sometimes it looks like 1 after "flattening" the batch), so kernel launch overhead accounts for a large portion of end-to-end latency.

**中文对照**：服务端"小批量/低延迟"很常见：批量非常小（有时在"扁平化"批量后看起来像 1），因此内核启动开销占端到端延迟的很大一部分。

Input token count (number of patches) varies frequently: Different image/video resolutions and different batch composition lead to different sequence lengths
S — and this is precisely the biggest obstacle for CUDA Graph (unstable shapes).

**中文对照**：输入令牌数量（patch 数量）变化频繁：不同的图像/视频分辨率和不同的批量组成导致不同的序列长度 S——而这正是 CUDA Graph 最大的障碍（不稳定的形状）。

The value of CUDA Graph: It captures a long sequence of GPU kernels with fixed shapes and fixed memory addresses into a graph; later, for the same shapes, it can replay the graph directly, dramatically reducing launch overhead and making GPU scheduling more compact.

This led us to seek a CUDA Graph enabled feature for ViT in order to improve ViT performance.

**中文对照**：CUDA Graph 的价值：它将具有固定形状和固定内存地址的 GPU 内核长序列捕获到一个图中；之后，对于相同的形状，它可以直接重放该图，显著减少启动开销并使 GPU 调度更加紧凑。

这促使我们为 ViT 寻求启用 CUDA Graph 的功能以提高 ViT 性能。

## Design and Restrictions

The new CUDA Graph enabled ViT logic is built on ViTCudaGraphRunner. This runner captures the "blocks + merger + deepstack merger (optional)" part of a vision transformer into a CUDA graph and replays it for identical shapes. See the following design consideration and restrictions for more details.

**中文对照**：## 设计和限制

新的启用 CUDA Graph 的 ViT 逻辑建立在 ViTCudaGraphRunner 之上。该运行器将视觉变换器的"块 + 合并器 + 深度堆叠合并器（可选）"部分捕获到 CUDA 图中，并为相同的形状重放它。请参阅以下设计注意事项和限制了解更多详情。

### Dynamic inputs to fit static constraints of CUDA Graph

Variable sequence length S is very common in ViT. While CUDA Graph requires fixed shapes. The solution is to build a graph cache by S(e.g., graph_key = S). The first time create a new S, and then capture a graph; afterwards, replay it.

If there are many distinct S values, we need to increase VRAM usage which is graph-private memory pools for many graphs.

**中文对照**：在 ViT 中，可变序列长度 S 非常常见。而 CUDA Graph 需要固定形状。解决方案是通过 S 构建图缓存（例如 graph_key = S）。第一次创建新的 S，然后捕获图；之后，重放它。

如果有许多不同的 S 值，我们需要增加 VRAM 使用量，这是许多图的图私有内存池。

### Stable addresses

Everything "parameter-like" becomes a static buffer:

- block_input / block_ws / block_output
- cu_full_len / cu_window_len and their kk variants
- sin_cos_ws

In this way to solve the underlying requirement: during replay, not allowed to swap tensors, can only modify tensor contents.

**中文对照**：所有"类似参数"的内容都变为静态缓冲区：

- block_input / block_ws / block_output
- cu_full_len / cu_window_len 及其 kk 变体
- sin_cos_ws

这样可以解决底层需求：在重放期间，不允许交换张量，只能修改张量内容。

### Attention backend arguments
Attention backend arguments are fixed inside the graph:

TritonAttn expects [cu_seqlens, cu_seqlens_kk, max_len]
FA3 expects [cu_seqlens, max_len]

max_len is frozen as an int constant.
cu_seqlens is cached into a dict during create_graph(), and its contents are not updated during subsequent replays.

For the same graph_key = S, you not only require the input shape to match, but also require the segmentation pattern in cu_seqlens (and window seqlens) to be identical. Otherwise, attention will segment the sequence incorrectly.

**中文对照**：注意力后端参数在图内部是固定的：

TritonAttn 期望 [cu_seqlens, cu_seqlens_kk, max_len]
FA3 期望 [cu_seqlens, max_len]

max_len 被冻结为整数常量。cu_seqlens 在 create_graph() 期间被缓存到字典中，其内容在后续重放期间不会更新。

对于相同的 graph_key = S，您不仅需要输入形状匹配，还需要 cu_seqlens（和窗口 seqlens）中的分段模式相同。否则，注意力将错误地分段序列。

### Rotary buffer management
The feature reallocates a larger sin_cos_ws when seq_len increases.
The max_content_len is used to make sure the maximum size of the allocated rotary buffer.

**中文对照**：### 旋转缓冲区管理
当 seq_len 增加时，该功能会重新分配更大的 sin_cos_ws。max_content_len 用于确保分配的旋转缓冲区的最大大小。


## Command Example
You can enable CUDA Graph for ViT by setting env variable `SGLANG_VIT_ENABLE_CUDA_GRAPH=1`, for example:
```
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct
```
Or you can run CUDA Graph for ViT together with Piecewise CUDA Graph feature by both setting env variable `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` and setting `--enable-piecewise-cuda-graph`, for example:
```
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --piecewise-cuda-graph-max-tokens 4096 \
  --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-compiler eager
```

## Known supported models
- Qwen2.5-VL (https://github.com/sgl-project/sglang/pull/14422)
- Qwen3-VL (https://github.com/sgl-project/sglang/pull/15320)

## 代码实现
- **核心文件**：`python/sglang/srt/multimodal/vit_cuda_graph_runner.py`
- **架构**：实现围绕 `ViTCudaGraphRunner` 展开，它为 Vision Transformer（ViT）编码器提供"记录-重放"机制。它将 GPU 内核序列（块和合并器）捕获到静态计算图中，以消除 CPU 到 GPU 的调度开销。
- **关键代码片段**：
  - `_create_graph(self, ...)`：使用 `torch.cuda.graph(graph)` 记录特定输入序列长度的前向传播
  - `replay(self, ...)`：更新输入缓冲区并使用 `self.block_graphs[graph_key].replay()` 执行预记录的图
  - `run(self, ...)`：入口点，检查当前输入形状是否存在图，如有必要创建新图，然后重放
- **集成要点**：通过环境变量 `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 启用。运行器与 ViT 模块实例化，并管理特定形状的缓存以处理不同的图像/视频分辨率。
