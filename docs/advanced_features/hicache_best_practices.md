# SGLang HiCache Best Practices

## Why HiCache Matters

SGLang HiCache extends the traditional RadixAttention with a three-tier hierarchical KV caching system that dramatically improves performance for long-context and multi-turn conversation scenarios. By intelligently managing KV caches across GPU memory, host memory, and external storage backends, HiCache addresses the fundamental capacity bottleneck that limits cache hit rates in conventional systems.

**中文对照**：# SGLang HiCache 最佳实践

## 为什么 HiCache 很重要

SGLang HiCache 通过三层分层 KV 缓存系统扩展了传统的 RadixAttention，极大地提高了长上下文和多轮对话场景的性能。通过智能管理 GPU 内存、主机内存和外部存储后端之间的 KV 缓存，HiCache 解决了限制传统系统中缓存命中率的根本容量瓶颈。

## Configuration Guidelines

## Core HiCache Parameters

```bash
# Essential HiCache flags
--page-size 64                        # Page size for cache management
--enable-hierarchical-cache           # Enable HiCache
--hicache-ratio 2                     # Host memory ratio (2x GPU memory)
--hicache-size 100                    # Host memory size in GBs, will override the above ratio
--hicache-io-backend kernel           # The I/O backend of moving data between CPU and GPU
--hicache-write-policy write_through  # Cache write policy from GPU to CPU
--hicache-storage-backend             # Optional storage backend (e.g., hf3fs, mooncake, etc.)
```

Notes:

- Besides configuring `--hicache-storage-backend` at startup, SGLang also supports **runtime attach/detach** of the HiCache storage backend (no restart required) via HTTP admin endpoints. See [Runtime Attach/Detach HiCache Storage Backend](hicache_storage_runtime_attach_detach.md).

## Key Configurations with Storage Backends Enabled

### Memory Layout Optimization

```bash
# Page-first: Optimized for I/O efficiency with zero-copy (recommended with kernel backend)
--hicache-mem-layout page_first
# Page-first-direct: Optimized for direct I/O operations (Compatible with fa3 and same zero-copy performance as page_first)
--hicache-mem-layout page_first_direct
# Layer-first
--hicache-mem-layout layer_first
```
**Layout Compatibility:**
- `page_first`: Only compatible with `kernel` I/O backend, automatically switches to `layer_first` with `direct` backend
- `page_first_direct`: Specifically designed for `direct` I/O backend with optimized memory organization

**中文对照**：注意：

- 除了在启动时配置 `--hicache-storage-backend` 之外，SGLang 还支持**运行时附加/分离** HiCache 存储后端（无需重启），通过 HTTP 管理端点。请参阅[运行时附加/分离 HiCache 存储后端](hicache_storage_runtime_attach_detach.md)。

## 启用存储后端时的关键配置

### 内存布局优化

```bash
# Page-first：针对 I/O 效率优化，零拷贝（与 kernel 后端配合推荐）
--hicache-mem-layout page_first
# Page-first-direct：针对直接 I/O 操作优化（与 fa3 兼容，与 page_first 具有相同的零拷贝性能）
--hicache-mem-layout page_first_direct
# Layer-first
--hicache-mem-layout layer_first
```
**布局兼容性：**
- `page_first`：仅与 `kernel` I/O 兼容，使用 `direct` 后端时自动切换到 `layer_first`
- `page_first_direct`：专为 `direct` I/O 后端设计，具有优化的内存组织

### Prefetch Policies

```bash
# Best-effort: Terminate prefetch when needed
--hicache-storage-prefetch-policy best_effort
# Wait-complete: Ensure complete prefetch, higher cache reuse
--hicache-storage-prefetch-policy wait_complete
# Timeout: Balance between completion and best-effort
--hicache-storage-prefetch-policy timeout
```

### Integration with PD Disaggregation

HiCache works seamlessly with PD Disaggregation. You can choose between two configurations:

1. **Prefill-only HiCache**: Enable HiCache only on Prefill nodes, allowing KV cache sharing among Prefill instances
2. **Full HiCache with async offloading**: Enable HiCache on Prefill nodes and async KV cache offloading on Decode nodes, allowing Prefill nodes to reuse KV caches from Decode nodes in multi-turn dialogue scenarios

**中文对照**：### 预取策略

```bash
# Best-effort：根据需要终止预取
--hicache-storage-prefetch-policy best_effort
# Wait-complete：确保完整预取，更高的缓存重用
--hicache-storage-prefetch-policy wait_complete
# Timeout：在完成和尽力之间取得平衡
--hicache-storage-prefetch-policy timeout
```

### 与 PD 分离集成

HiCache 与 PD 分离无缝协作。您可以选择两种配置之一：

1. **仅预填充 HiCache**：仅在预填充节点上启用 HiCache，允许预填充实例之间共享 KV 缓存
2. **完整 HiCache 异步卸载**：在预填充节点上启用 HiCache，并在解码节点上启用异步 KV 缓存卸载，允许预填充节点在多轮对话场景中重用来自解码节点的 KV 缓存

```bash
# Prefill node with HiCache enabled for cross-prefill sharing (ideal for SystemPrompt scenarios)
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake

# Decode node with async offloading enabled for KV cache reuse by Prefill (ideal for multi-turn conversations)
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --page-size 64 \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-decode-enable-offload-kvcache \  # Enable async KV cache offloading in decode node
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake
```


### Deployment with HF3FS

Here is an example of deploying DeepSeek-R1 with HiCache-HF3FS. For more details, see the [HF3FS Documentation](../../python/sglang/srt/mem_cache/storage/hf3fs/docs/README.md).

**中文对照**：## 使用 HF3FS 部署

以下是使用 HiCache-HF3FS 部署 DeepSeek-R1 的示例。更多详情，请参阅 [HF3FS 文档](../../python/sglang/srt/mem_cache/storage/hf3fs/docs/README.md)。

```bash
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --log-level info \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
```

### Deployment with Mooncake

Here is an example of deploying Qwen3-235B-A22B-Instruct-2507 with Mooncake. For more details, see the [Mooncake Documentation](../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md).

**中文对照**：## 使用 Mooncake 部署

以下是使用 Mooncake 部署 Qwen3-235B-A22B-Instruct-2507 的示例。更多详情，请参阅 [Mooncake 文档](../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md)。

```bash
# Set Mooncake environment variables
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=816043786240
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="$DEVICE_LIST"
export MOONCAKE_MASTER=127.0.0.1:50051

# Launch SGLang server with Mooncake backend
python3 -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --tp 8 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout
```


## Custom Storage Backend Integration

To integrate a new storage backend:

1. **Implement three core methods:**
   - `get(key)`: Retrieve value by key
   - `exists(key)`: Check key existence
   - `set(key, value)`: Store key-value pair

2. **Register your backend:** Add your storage backend to the HiCache [BackendFactory](../../python/sglang/srt/mem_cache/storage/backend_factory.py#L188)

The HiCache controller handles all scheduling and synchronization automatically.

### Dynamic Backend Loading

Alternatively, you can use dynamic loading to avoid hard-coding your backend in the repository:

**中文对照**：## 自定义存储后端集成

要集成新的存储后端：

1. **实现三个核心方法：**
   - `get(key)`：按键检索值
   - `exists(key)`：检查键是否存在
   - `set(key, value)`：存储键值对

2. **注册您的后端：** 将您的存储后端添加到 HiCache [BackendFactory](../../python/sglang/srt/mem_cache/storage/backend_factory.py#L188)

HiCache 控制器自动处理所有调度和同步。

### 动态后端加载

或者，您可以使用动态加载以避免在仓库中硬编码您的后端：

```bash
python3 -m sglang.launch_server \
  --model-path your-model \
  --enable-hierarchical-cache \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{"backend_name":"custom_backend_name", "module_path": "your_module_path", "class_name": "YourHiCacheClassName"}'
```

**Configuration Parameters:**
- `--hicache-storage-backend`: Set to `dynamic`
- `--hicache-storage-backend-extra-config`: JSON configuration with:
  - `backend_name`: Custom backend identifier
  - `module_path`: Python module path to your implementation
  - `class_name`: Your HiCache implementation class name
  - `interface_v1`: 0 (disable) or 1 (enable) to control usage of batch_get_v1 and batch_set_v1 methods

**中文对照**：**配置参数：**
- `--hicache-storage-backend`：设置为 `dynamic`
- `--hicache-storage-backend-extra-config`：JSON 配置，包含：
  - `backend_name`：自定义后端标识符
  - `module_path`：您的实现的 Python 模块路径
  - `class_name`：您的 HiCache 实现类名
  - `interface_v1`：0（禁用）或 1（启用）以控制 batch_get_v1 和 batch_set_v1 方法的使用


## 代码实现

### 核心文件
- `python/sglang/srt/mem_cache/hiradix_cache.py`: 分层radix attention的核心逻辑。
- `python/sglang/srt/server_args.py`: 定义配置的命令行参数。
- `python/sglang/srt/mem_cache/storage/`: 各种存储后端的实现（hf3fs、mooncake等）。

### 架构
HiCache性能由定义KV缓存跨层级分布的参数控制。`HiRadixCache` 管理GPU内存（L1）、主机内存（L2）和外部存储（L3）之间的数据移动。`ServerArgs` 处理像 `--hicache-ratio` 这样的参数以分配适当的内存缓冲区，而预取策略决定何时从较低层级向GPU获取数据。

### 关键代码逻辑
`server_args.py` 中的参数处理：
```python
# server_args.py
parser.add_argument("--hicache-ratio", type=float, default=2.0)
parser.add_argument("--hicache-storage-prefetch-policy", type=str,
                    choices=["best_effort", "wait_complete", "timeout"])
```
`hiradix_cache.py` 中的分层查找逻辑：
```python
# hiradix_cache.py
def match_prefix(self, token_ids):
    # 1. Search in GPU (L1)
    # 2. Search in CPU (L2) / Storage (L3)
    # 3. Initiate prefetch if hit in lower tiers
```

### 集成要点
HiCache与 `RadixCache` 集成以扩展容量。它挂接到调度循环中，确保在请求分派到GPU之前预取所需的KV块，从而最小化由于高层级缓存未命中导致的延迟停顿。

## Community and Support

- **GitHub Issues**: Report bugs and feature requests
- **Slack Channel**: Join community discussions in #sgl-kv-cache-store
- **Documentation**: Refer to storage backend-specific guides

**中文对照**：## 社区和支持

- **GitHub Issues**：报告错误和功能请求
- **Slack 频道**：在 #sgl-kv-cache-store 中加入社区讨论
- **文档**：参阅特定于存储后端的指南

---

*This document will be continuously updated based on community feedback and new features. Contributions and suggestions are welcome!*
