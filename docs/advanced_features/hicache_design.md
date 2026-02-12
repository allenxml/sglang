# HiCache System Design and Optimization

This document provides a comprehensive overview of SGLang HiCache, covering its system architecture, workflow and key components. It also details configuration parameters, optimization techniques, and integration with various L3 storage backends, serving as a complete reference for users and developers to understand and tune HiCache for efficient LLM inference.

## Why and What is HiCache?

In large language model inference, the prefill phase is often time-consuming: input sequences need to be first converted into Key-Value cache (KV cache) for subsequent decoding. When multiple requests share the same prefix, the KV cache for that prefix is identical. By caching and reusing these shared KV caches, redundant computation can be avoided. To address this, SGLang introduced RadixAttention, which leverages idle GPU memory to cache and reuse prefix KV caches, and **HiCache**, which extends this idea to host memory and distributed storage.

Inspired by the classic three-level cache design of modern CPUs, HiCache organizes GPU memory as L1, host memory as L2, and distributed storage as L3. This hierarchy enables HiCache to fully exploit the "idle" storage space of GPUs and CPUs, while integrating distributed cache systems such as Mooncake, 3FS, NIXL, and AIBrix KVCache for global KV cache storage and scheduling. As a result, HiCache significantly expands KV cache capacity while maintaining strong read performance—especially in workloads such as multi-QA and long-context inference, where KV cache reuse is frequent. For detailed benchmark results, see [this blog](https://lmsys.org/blog/2025-09-10-sglang-hicache/).

**中文对照**：# HiCache 系统设计和优化

本文档提供了 SGLang HiCache 的全面概述，包括其系统架构、工作流程和关键组件。它还详细介绍了配置参数、优化技术以及与各种 L3 存储后端的集成，作为用户和开发者的完整参考，以理解和调优 HiCache 以实现高效的 LLM 推理。

## 为什么以及什么是 HiCache？

在大语言模型推理中，预填充阶段通常很耗时：输入序列需要首先转换为键值缓存（KV 缓存）以供后续解码。当多个请求共享相同前缀时，该前缀的 KV 缓存是相同的。通过缓存和重用这些共享的 KV 缓存，可以避免冗余计算。为了解决这个问题，SGLang 引入了 RadixAttention，它利用空闲 GPU 内存来缓存和重用前缀 KV 缓存，以及 **HiCache**，它将这个想法扩展到主机内存和分布式存储。

受现代 CPU 经典三级缓存设计的启发，HiCache 将 GPU 内存组织为 L1，主机内存为 L2，分布式存储为 L3。这种层次结构使 HiCache 能够充分利用 GPU 和 CPU 的"空闲"存储空间，同时集成 Mooncake、3FS、NIXL 和 AIBrix KVCache 等分布式缓存系统，用于全局 KV 缓存存储和调度。因此，HiCache 显著扩展了 KV 缓存容量，同时保持强大的读取性能——特别是在多 QA 和长上下文推理等工作负载中，KV 缓存重用很频繁。详细的基准测试结果，请参阅[这篇博客](https://lmsys.org/blog/2025-09-10-sglang-hicache/)。


## System Design

### Overall Architecture

In many modern CPU architectures, the small but fast L1 and L2 caches are private to each core, enabling rapid access to the hottest data, while the larger L3 cache is shared across all cores to significantly reduce redundancy within the cache. Similarly, in HiCache, the L1 and L2 KV caches are private to each inference instance, whereas the L3 KV cache is shared among all inference instances within the cluster.

**中文对照**：## 系统设计

### 整体架构

在许多现代 CPU 架构中，小而快的 L1 和 L2 缓存对每个核心都是专用的，能够快速访问最热的数据，而较大的 L3 缓存在所有核心之间共享，以显著减少缓存内的冗余。类似地，在 HiCache 中，L1 和 L2 KV 缓存对每个推理实例都是专用的，而 L3 KV 缓存在集群内的所有推理实例之间共享。

### HiRadixTree: Metadata Organization in HiCache

For KV cache data organization, HiCache builds upon the RadixTree structure introduced in RadixAttention and proposes HiRadixTree. In RadixAttention, each node of the RadixTree corresponds to the KV cache of a consecutive span of tokens in GPU memory. A path from the root to a leaf node represents the prefix of a request, and shared prefixes across multiple requests can reuse the same nodes, thereby avoiding redundant storage.

HiRadixTree extends this idea: each node corresponds to the KV cache of a span of consecutive tokens and records where that KV cache is stored—whether in local GPU memory, CPU memory, L3 storage, or multiple of these tiers. If stored locally, HiRadixTree maintains precise metadata, including the exact storage address. However, to reduce overhead, HiRadixTree does not store or continuously synchronize metadata for L3 KV cache. Instead, when accessing L3 data, it queries the backend in real time to retrieve the necessary metadata, such as whether the data exists and on which server and location it resides.

**中文对照**：### HiRadixTree：HiCache 中的元数据组织

对于 KV 缓存数据组织，HiCache 基于 RadixAttention 中引入的 RadixTree 结构，并提出了 HiRadixTree。在 RadixAttention 中，RadixTree 的每个节点对应 GPU 内存中连续令牌跨度的 KV 缓存。从根到叶节点的路径表示请求的前缀，多个请求之间的共享前缀可以重用相同的节点，从而避免冗余存储。

HiRadixTree 扩展了这个想法：每个节点对应连续令牌跨度的 KV 缓存，并记录该 KV 缓存存储在哪里——无论是在本地 GPU 内存、CPU 内存、L3 存储还是这些层的多个层中。如果存储在本地，HiRadixTree 会维护精确的元数据，包括确切的存储地址。然而，为了减少开销，HiRadixTree 不存储或持续同步 L3 KV 缓存的元数据。相反，在访问 L3 数据时，它会实时查询后端以检索必要的元数据，例如数据是否存在以及它位于哪个服务器和位置。

### Overall Workflow

The workflow of HiCache mainly involves three key operations: **local match**, **prefetch** and **write-back**. When the system receives a new request, it first searches the local L1 and L2 caches for matching KV caches. For parts not found locally, it attempts to prefetch from L3. After prefetching, all required KV caches are loaded into the GPU for computation. Once the prefill computation is complete, the system considers storing the newly generated data into L2 or L3.

**中文对照**：### 整体工作流程

HiCache 的工作流程主要涉及三个关键操作：**本地匹配**、**预取**和**回写**。当系统收到新请求时，它首先在本地 L1 和 L2 缓存中搜索匹配的 KV 缓存。对于本地找不到的部分，它尝试从 L3 预取。预取后，所有需要的 KV 缓存都被加载到 GPU 中进行计算。一旦预填充计算完成，系统就会考虑将新生成的数据存储到 L2 或 L3 中。

![HiCache Workflow](https://lmsys.org/images/blog/hicache/hicache_overview.png)

### Local Match

Local matching is the first step in HiCache's workflow, where incoming request tokens are matched against the HiRadixTree to locate cached KV data in local memory tiers (L1 GPU memory and L2 host memory).

The matching algorithm traverses the HiRadixTree from the root node, following child nodes that match the token sequence prefix. At each node, the incoming token sequence is compared with the node's stored token sequence. When `page_size > 1`, matching is performed at the page granularity to optimize memory access patterns. If a match terminates within a node's stored sequence, the node is automatically split to create an exact boundary, improving the efficiency of future matches.

The algorithm returns a continuous prefix of the request, with the first part residing in L1 and the latter part in L2.

Since the process only requires traversing the local HiRadixTree and does not involve any actual data copying, local matching is extremely fast.

**中文对照**：### 本地匹配

本地匹配是 HiCache 工作流程的第一步，传入的请求令牌与 HiRadixTree 匹配，以在本地内存层（L1 GPU 内存和 L2 主机内存）中定位缓存的 KV 数据。

匹配算法从根节点遍历 HiRadixTree， following 匹配令牌序列前缀的子节点。在每个节点，传入的令牌序列与节点存储的令牌序列进行比较。当 `page_size > 1` 时，匹配以页面粒度执行以优化内存访问模式。如果匹配在节点存储的序列内终止，则节点会自动拆分以创建精确边界，提高未来匹配的效率。

算法返回请求的连续前缀，第一部分位于 L1，后一部分位于 L2。

由于该过程仅需要遍历本地 HiRadixTree 且不涉及任何实际数据复制，本地匹配非常快。

### Prefetch from L3

Data prefetching is one of HiCache's core optimization techniques, designed to proactively load KV caches from L3 storage into local L2 memory, thereby reducing access latency during subsequent operations.

**Prefetch Trigger Conditions**:
After local matching, for the parts not found in L1 or L2, the system queries L3 to retrieve metadata for the next continuous matching KV caches. If the length of hit cache in L3 exceeds a threshold (default: 256 tokens, configurable), a prefetch operation is triggered.

**Prefetch Strategies**: HiCache provides three different prefetch termination strategies to address different scenario needs:
- **best_effort**: Terminates immediately when GPU can execute prefill computation, with no waiting time, suitable for scenarios extremely sensitive to latency.
- **wait_complete**: Must wait for all prefetch operations to complete, suitable for scenarios requiring high cache hit rates.
- **timeout**: Terminates after specified time or when complete, balancing latency and cache hit rate needs.

After prefetching stops, the data already fetched is used together with the local data for the prefill computation.

For **timeout** strategy, HiCache introduces two configuration parameters to support fine-grained control over prefetch timeout conditions:

* `prefetch_timeout_base`: the base timeout, representing overhead unrelated to the number of tokens (e.g., scheduling and synchronization).
* `prefetch_timeout_per_ki_token`: the incremental timeout per thousand tokens.

The timeout is computed as:

```
timeout = prefetch_timeout_base + prefetch_timeout_per_ki_token * num_token_to_fetch / 1024
```

**中文对照**：### 从 L3 预取

数据预取是 HiCache 的核心优化技术之一，旨在主动将 KV 缓存从 L3 存储加载到本地 L2 内存，从而减少后续操作期间的访问延迟。

**预取触发条件**：
在本地匹配之后，对于在 L1 或 L2 中找不到的部分，系统查询 L3 以检索下一个连续匹配的 KV 缓存的元数据。如果 L3 中命中缓存的长度超过阈值（默认值：256 个令牌，可配置），则触发预取操作。

**预取策略**：HiCache 提供三种不同的预取终止策略以满足不同场景需求：
- **best_effort**：当 GPU 可以执行预填充计算时立即终止，无需等待时间，适用于对延迟极其敏感的场景。
- **wait_complete**：必须等待所有预取操作完成，适用于需要高缓存命中率的场景。
- **timeout**：在指定时间后或完成后终止，平衡延迟和缓存命中率需求。

预取停止后，已获取的数据与本地数据一起用于预填充计算。

对于 **timeout** 策略，HiCache 引入了两个配置参数以支持对预取超时条件的细粒度控制：

* `prefetch_timeout_base`：基础超时，代表与令牌数量无关的开销（例如，调度和同步）。
* `prefetch_timeout_per_ki_token`：每千令牌的增量超时。

超时计算如下：

```
timeout = prefetch_timeout_base + prefetch_timeout_per_ki_token * num_token_to_fetch / 1024
```

### Data Write-back

The write-back mechanism is responsible for moving frequently accessed KV caches from L1 to L2 and L3, enabling larger and longer-term storage as well as cache sharing across instances.

**Configurable Write-back Policies**: HiCache supports three write-back strategies:

* **write_through**: Every access is immediately written back to the next level. When bandwidth is sufficient, this strategy provides the strongest caching benefit.
* **write_through_selective**: Data is written back only after the access frequency exceeds a threshold. This strategy backs up only hot data, reducing I/O overhead.
* **write_back**: Data is written back to the next level only when it is evicted from the upper level. This strategy alleviates storage pressure and is suitable for scenarios where storage capacity is limited but memory utilization must be maximized.

**Cross-instance Sharing**: When data is written back from L2 to L3, only data not already present in L3 is transferred. KV caches stored in L3 can then be shared across all SGLang instances in the cluster (depending on the L3 backend implementation), significantly improving cache hit rates within the same memory budget.

**中文对照**：### 数据回写

回写机制负责将频繁访问的 KV 缓存从 L1 移动到 L2 和 L3，实现更大和更长期的存储以及跨实例的缓存共享。

**可配置的回写策略**：HiCache 支持三种回写策略：

* **write_through**：每次访问都立即写回到下一层。当带宽充足时，此策略提供最强的缓存收益。
* **write_through_selective**：数据仅在访问频率超过阈值后才回写。此策略仅备份热数据，减少 I/O 开销。
* **write_back**：数据仅在从上层逐出时才回写到下一层。此策略缓解存储压力，适用于存储容量有限但必须最大化内存利用的场景。

**跨实例共享**：当数据从 L2 回写到 L3 时，仅传输 L3 中尚不存在的数据。存储在 L3 中的 KV 缓存随后可以在集群中的所有 SGLang 实例之间共享（取决于 L3 后端实现），在相同内存预算内显著提高缓存命中率。

### Multi-Rank Synchronization

During multi-GPU parallel computation, such as tensor parallelism (TP), HiCache must ensure consistent states across different ranks. Therefore, critical computation steps require the use of `all_reduce` for state synchronization.

For example, during prefetching, `all_reduce(op=min)` is used to ensure that all ranks obtain the same number of L3 hits, preventing inconsistent judgments about whether the prefetch threshold has been reached. Similarly, after prefetching completes or terminates, `all_reduce(op=min)` is again required to guarantee consensus among ranks on the prefix length of the successfully retrieved KV cache.

**中文对照**：### 多 Rank 同步

在多 GPU 并行计算（如张量并行（TP））期间，HiCache 必须确保不同 rank 之间的状态一致。因此，关键的计算步骤需要使用 `all_reduce` 进行状态同步。

例如，在预取期间，使用 `all_reduce(op=min)` 来确保所有 rank 获得相同数量的 L3 命中，防止关于是否达到预取阈值的判断不一致。类似地，在预取完成或终止后，再次需要 `all_reduce(op=min)` 以保证各 rank 对成功检索的 KV 缓存的前缀长度达成共识。

### Data Transfer Optimization

**Zero-Copy Data Transfers**: Both prefetching and write-back involve substantial data movement. Minimizing the number of data copies can significantly improve system performance. HiCache supports passing memory addresses and sizes directly when transferring data from L2 memory to an L3 backend.

**"Batch-Oriented" Data Organization**: The granularity of data reads and writes has a major impact on performance. To address this, HiCache L3 stores and transfers KV cache data at the granularity of **pages** and supports different data layouts beyond the existing `layer first` scheme, including `page first` and `page first direct`. Under the `page first` and `page first direct` layouts, all KV cache data belonging to the same page is placed in contiguous memory, allowing it to be passed as a single object to L3 using zero-copy transfers.

**中文对照**：### 数据传输优化

**零拷贝数据传输**：预取和回写都涉及大量数据移动。最小化数据复制次数可以显著提高系统性能。HiCache 支持在从 L2 内存传输数据到 L3 后端时直接传递内存地址和大小。

**"面向批次"的数据组织**：数据读写粒度对性能有重大影响。为了解决这个问题，HiCache L3 以**页面**粒度存储和传输 KV 缓存数据，并支持除现有 `layer first` 方案之外的不同数据布局，包括 `page first` 和 `page first direct`。在 `page first` 和 `page first direct` 布局下，属于同一页的所有 KV 缓存数据都放置在连续内存中，允许使用零拷贝传输作为单个对象传递到 L3。

![HiCache L2 MEM layout](https://lmsys.org/images/blog/hicache/hicache_layout.png)

However, because GPU KV computation is naturally performed layer by layer, the GPU inherently operates in a `layer first` layout. When transferring `page first` data from L2 to the GPU, data must be transferred at the granularity of one token per layer. The `page first direct` layout mitigates this issue by grouping together all tokens of a given layer within a page, allowing transfers from L2 to GPU to be aggregated at the page-layer level.

**CPU-to-GPU Transfer Optimizations**: In HiCache, moving data from CPU memory to GPU is as performance-critical as prefetching data from L3 to L2. HiCache employs several optimizations for this process:

* **Compute-Transfer Overlap**: During the prefill phase, when transferring data from CPU to GPU, HiCache overlaps layers by concurrently loading the KV cache of layer N+1 while computing layer N. This effectively hides data transfer latency.
* **GPU-assisted I/O Kernels**: On top of `cudaMemcpyAsync`, HiCache implements a set of GPU-assisted I/O kernels specifically optimized for KV cache transfers between CPU and GPU. Compared to the baseline approach, these kernels achieve up to 3x higher transfer speed.

**Write-back Optimization for MLA**: For MHA (Multi-Head Attention) models under multi-TP, each rank holds `1/tp_size` of a token's KV data. In contrast, for MLA (Multi-Layer Attention) models, all ranks hold the complete and identical KV data for each token. HiCache includes a dedicated optimization for MLA: only one rank initiates the write-back operation, ensuring that data is not redundantly stored across ranks.

**中文对照**：然而，由于 GPU KV 计算自然按层执行，GPU 本质上以 `layer first` 布局运行。当从 L2 向 GPU 传输 `page first` 数据时，数据必须按每层一个令牌的粒度传输。`page first direct` 布局通过将页面内给定层的所有令牌分组在一起来缓解此问题，允许在页面层级别聚合从 L2 到 GPU 的传输。

**CPU 到 GPU 传输优化**：在 HiCache 中，将数据从 CPU 内存移动到 GPU 与从 L3 预取数据到 L2 一样对性能至关重要。HiCache 对此过程采用了多项优化：

* **计算-传输重叠**：在预填充阶段，当从 CPU 向 GPU 传输数据时，HiCache 通过在计算第 N 层的同时并行加载第 N+1 层的 KV 缓存来重叠层。这有效地隐藏了数据传输延迟。
* **GPU 辅助 I/O 内核**：在 `cudaMemcpyAsync` 之上，HiCache 实现了一组专门针对 CPU 和 GPU 之间 KV 缓存传输优化的 GPU 辅助 I/O 内核。与基线方法相比，这些内核实现了高达 3 倍的传输速度提升。

**MLA 回写优化**：对于多 TP 下的 MHA（多头注意力）模型，每个 rank 持有令牌 KV 数据的 `1/tp_size`。相比之下，对于 MLA（多层注意力）模型，所有 rank 持有每个令牌的完整且相同的 KV 数据。HiCache 包含针对 MLA 的专用优化：只有一个 rank 启动回写操作，确保数据不会在 rank 之间冗余存储。

### Integration with PD-Disaggregation Deployment Mode

SGLang supports a PD (Prefill-Decode) disaggregation deployment mode through the Mooncake TransferEngine (for details, see [this doc](https://docs.sglang.io/advanced_features/pd_disaggregation.html)). In the PD-disaggregation deployment mode, HiCache can be enabled on both the prefill nodes and decode nodes to optimize prefill performance. If enabled on decode nodes, the decode output will also be written back to L3.

### Unified Interfaces and Rich L3 Storage Backends

HiCache encapsulates all read, write, and query operations on L3 backends within the `class HiCacheStorage(ABC)`, exposing a set of simple and consistent interfaces. This design supports a wide range of L3 storage backends and allows users to select the one that best fits their specific use cases.

- **Mooncake**: Mooncake is a high-performance caching system for LLM inference that leverages RDMA and multi-NIC resources to enable zero-copy, ultra-fast data transfers. Try Mooncake [here](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/mooncake_store).

- **DeepSeek 3FS (HF3FS)**: HF3FS is a Kubernetes-native distributed storage solution with operator-based deployment. Try HF3FS [here](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/hf3fs).

- **NIXL**: NIXL provides a unified API for accessing various storage plugins, including but not limited to DeepSeek's 3FS, GPU Direct Storage (GDS) and Amazon S3-compatible object storage. Try NIXL [here](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/nixl).

- **AIBrix KVCache**: AIBrix KVCache is a production-ready KVCache Offloading Framework, which enables efficient memory tiering and low-overhead cross-engine reuse. Try AIBrix KVCache [here](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/aibrix_kvcache).

- **HiCacheFile**: A simple file-based storage backend for demonstration purposes.

Specifically, **LMCache**, an efficient KV cache layer for enterprise-scale LLM inference, provides an alternative solution to HiCache. Try LMCache [here](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/lmcache).

**中文对照**：### 与 PD 分离部署模式的集成

SGLang 通过 Mooncake TransferEngine 支持 PD（预填充-解码）分离部署模式（详情请参阅[此文档](https://docs.sglang.io/advanced_features/pd_disaggregation.html)）。在 PD 分离部署模式下，可以在预填充节点和解码节点上都启用 HiCache 以优化预填充性能。如果在解码节点上启用，解码输出也将回写到 L3。

### 统一的接口和丰富的 L3 存储后端

HiCache 将 L3 后端上的所有读取、写入和查询操作封装在 `class HiCacheStorage(ABC)` 中，暴露出一套简单一致的接口。此设计支持广泛的 L3 存储后端，并允许用户选择最适合其特定用例的后端。

- **Mooncake**：Mooncake 是一个用于 LLM 推理的高性能缓存系统，利用 RDMA 和多 NIC 资源实现零拷贝、超快速数据传输。在[此处](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/mooncake_store)尝试 Mooncake。

- **DeepSeek 3FS (HF3FS)**：HF3FS 是一个 Kubernetes 原生分布式存储解决方案，具有基于操作员的部署。在[此处](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/hf3fs)尝试 HF3FS。

- **NIXL**：NIXL 提供统一的 API 来访问各种存储插件，包括但不限于 DeepSeek 的 3FS、GPU Direct Storage (GDS) 和 Amazon S3 兼容的对象存储。在[此处](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/nixl)尝试 NIXL。

- **AIBrix KVCache**：AIBrix KVCache 是一个生产级的 KVCache 卸载框架，实现了高效的内存分层和低开销的跨引擎重用。在[此处](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/aibrix_kvcache)尝试 AIBrix KVCache。

- **HiCacheFile**：一个简单的基于文件的存储后端，用于演示目的。

具体来说，**LMCache** 是一个用于企业级 LLM 推理的高效 KV 缓存层，提供了 HiCache 的替代解决方案。在[此处](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/lmcache)尝试 LMCache。

## Related Parameters

- **`--enable-hierarchical-cache`**: Enable hierarchical cache functionality. This is required to use HiCache.

- **`--hicache-ratio HICACHE_RATIO`**: The ratio of the size of host KV cache memory pool to the size of device pool. For example, a value of 2 means the host memory pool is twice as large as the device memory pool. The value of this parameter must be greater than 1, as the current implementation requires the host memory allocated for the KV cache to be larger than the device memory allocated for the KV cache.

- **`--hicache-size HICACHE_SIZE`**: The size of host KV cache memory pool in gigabytes. This parameter overrides `hicache-ratio` if set. For example, `--hicache-size 30` allocates 30GB (1GB = 1e9 bytes) for the host memory pool **for each rank**. If there are 8 ranks, then the total memory size is 240GB. Just like `hicache-ratio`, the value of this parameter must be larger than the size of device memory allocated for KV cache.

**Note**: `--hicache-ratio` and `--hicache-size` are two critical parameters. In general, a larger HiCache size leads to a higher cache hit rate, which improves prefill performance. However, the relationship between cache size and hit rate is not linear. Once most reusable KV data—especially hot tokens—are already cached, further increasing the size may yield only marginal performance gains. Users can set these parameters based on their workload characteristics and performance requirements.

- **`--page-size PAGE_SIZE`**: The number of tokens per page. This parameter determines the granularity of KV cache storage and retrieval. Larger page sizes reduce metadata overhead and improve I/O efficiency for storage backends, but may lower the cache hit rate when only part of a page matches the stored KV cache. For workloads with long common prefixes, larger pages can improve performance, while workloads with more diverse prefixes may benefit from smaller pages. See [Data Transfer Optimization](#data-transfer-optimization) for how page granularity affects I/O performance.

- **`--hicache-storage-prefetch-policy {best_effort,wait_complete,timeout}`**: Controls when prefetching from storage should stop. See [Prefetch from L3](#prefetch-from-l3) for details.
  - `best_effort`: Prefetch as much as possible without blocking
  - `wait_complete`: Wait for prefetch to complete before proceeding
  - `timeout`: Terminates after specified time or when complete (Recommended for production environments, as setting an appropriate timeout helps the system meet required SLOs)

- **`--hicache-write-policy {write_back,write_through,write_through_selective}`**: Controls how data is written from faster to slower memory tiers. See [Data Write-back](#data-write-back) for details.
  - `write_through`: Immediately writes data to all tiers (strongest caching benefits)
  - `write_through_selective`: Uses hit-count tracking to back up only frequently accessed data
  - `write_back`: Writes data back to slower tiers only when eviction is needed (reduces I/O load)

- **`--hicache-io-backend {direct,kernel}`**: Choose the I/O backend for KV cache transfer between CPU and GPU. See [Data Transfer Optimization](#data-transfer-optimization) for details.
  - `direct`: Standard CUDA memory copy operations
  - `kernel`: GPU-assisted I/O kernels (recommended for better performance)

- **`--hicache-mem-layout {layer_first,page_first,page_first_direct}`**: Memory layout for the host memory pool. See [Data Transfer Optimization](#data-transfer-optimization) for details.
  - `layer_first`: Compatible with GPU computation kernels (default for GPU memory)
  - `page_first`: Optimized for I/O efficiency
  - `page_first_direct`: Groups all tokens of a given layer within a page, allowing transfers from L2 to GPU to be aggregated at the page-layer level

- **`--hicache-storage-backend {file,mooncake,hf3fs,nixl,aibrix,dynamic}`**: Choose the storage backend for the L3 tier. Built-in backends: file, mooncake, hf3fs, nixl, aibrix. For dynamic backend, use --hicache-storage-backend-extra-config to specify: `backend_name` (custom name), `module_path` (Python module path), `class_name` (backend class name). See [Unified Interfaces and Rich L3 Storage Backends](#unified-interfaces-and-rich-l3-storage-backends) for available backends.

- **`--enable-lmcache`**: Using LMCache as an alternative hierarchical cache solution.

- **`--hicache-storage-backend-extra-config HICACHE_STORAGE_BACKEND_EXTRA_CONFIG`**: the extra config can be either
  - a JSON string containing extra configuration for the storage backend, e.g., `--hicache-storage-backend-extra-config '{"prefetch_threshold":512, "prefetch_timeout_base": 0.5, "prefetch_timeout_per_ki_token": 0.25}' `, or
  - a TOML or JSON or YAML file specifying the extra configuration for the storage backend (to differentiate from the JSON string input, prepend a `@` in front of the file name), e.g., `--hicache-storage-backend-extra-config "@config.toml"` where `config.toml` is the config file containing the complex configurations. This can be useful when the configuration consists of many or complex key-value pairs (for instance, it is preferred to use a config file for NIXL backend as its configurations can be complex).

## 代码实现

### 核心文件

- [hiradix_cache.py](../../python/sglang/srt/mem_cache/hiradix_cache.py): 实现 `HiRadixCache` 类，扩展标准 RadixCache 以管理跨多个内存层级的元数据。它编排本地匹配、预取和与存储控制器协调的高层逻辑。
- [hicache_storage.py](../../python/sglang/srt/mem_cache/hicache_storage.py): 定义 `HiCacheStorage` 抽象基类，为 L3 存储操作提供统一接口。此抽象允许系统支持多种后端，同时暴露一致的 `read`、`write` 和 `query` 方法。
- [memory_pool_host.py](../../python/sglang/srt/mem_cache/memory_pool_host.py): 管理 L2 主机内存池，处理不同模型架构（MHA、MLA、NSA）的 CPU 内存分配。它支持 `page_first` 和 `page_first_direct` 等多种内存布局以优化 I/O 性能。
- [storage/](../../python/sglang/srt/mem_cache/storage/): 包含特定 L3 后端实现，如 `mooncake_store.py`、`storage_hf3fs.py`、`hicache_nixl.py` 和 `aibrix_kvcache_storage.py`。这些文件实现了分布式 KV 缓存访问的存储特定逻辑。

### 架构

实现遵循分层数据流：
1.  **L1 (GPU) ↔ L2 (Host)**: 通过 `HiCacheController` 管理，使用优化的 CUDA 内核或 `cudaMemcpyAsync`。数据在写回期间移动到 L2，在预填充期间检索。
2.  **L2 (Host) ↔ L3 (Remote)**: 存储控制器中的异步线程处理主机内存和分布式存储之间的数据移动。
3.  **预取/写回工作流**: 当请求到达时，系统在 L1/L2 中执行本地匹配。如果不足，则触发 L3 查询；命中的数据被预取到 L2 主机内存。预填充后，`HiCacheController` 根据 `write_policy` 异步将 GPU KV 缓存写回 L2/L3。

### 关键代码逻辑

- **本地匹配**: `hiradix_cache.py` 中的 `match_prefix` 方法覆盖基础实现，以识别 GPU 和 CPU 内存池中的匹配段，根据需要在页边界处拆分节点。
- **预取**: 预取逻辑在 `hiradix_cache.py` 中通过 `_prefetch_from_storage`（在请求处理期间调用）启动，它将 `PrefetchOperation` 任务提交给 `HiCacheController` 以进行异步执行。
- **写回**: 从 L1 到较慢层级的数据移动由 `hiradix_cache.py` 中的 `_write_back_to_storage` 处理，它识别可驱逐或"热"节点并将它们推送到控制器的备份队列。

### 集成要点

- **配置**: 所有 HiCache 参数（例如 `--enable-hierarchical-cache`、`--hicache-ratio`）都在 `server_args.py` 中定义，用于初始化 `HiRadixCache` 的 `CacheInitParams`。
- **调度器交互**: `Scheduler` 通过标准 `RadixCache` 接口与 `HiRadixCache` 交互。分层复杂性被隐藏，调度器接收关于可用前缀长度的信息，这些信息包括来自所有层级的缓存数据。
- **多 Rank 同步**: 在张量并行设置中，`HiRadixCache` 使用带有 `min` 运算符的 `torch.distributed.all_reduce` 来同步前缀匹配结果，并确保所有 rank 在执行前就预取长度达成一致。
