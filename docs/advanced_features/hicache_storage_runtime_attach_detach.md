# Runtime Attach/Detach HiCache Storage Backend (No Restart)

This document explains how to **dynamically attach/detach the HiCache L3 storage backend at runtime** (e.g., `mooncake` / `hf3fs` / `nixl` / `file` / `aibrix` / `eic`) while **SGLang is already running and serving traffic**, without restarting the process.

For safety and consistency, the current implementation **strictly requires** these operations to happen only when the service is **idle**:

- **No running requests**
- **No waiting/queued requests**

If the idle condition is not met, the API will fail fast (HTTP 400) and **will not modify** the current service state.

**中文对照**：# 运行时附加/分离 HiCache 存储后端（无需重启）

本文档解释了如何在 **SGLang 已经在运行并服务流量时**（例如 `mooncake` / `hf3fs` / `nixl` / `file` / `aibrix` / `eic`）**动态附加/分离 HiCache L3 存储后端**，而无需重启进程。

为了安全性和一致性，当前实现**严格要求**这些操作仅在服务**空闲**时发生：

- **没有正在运行的请求**
- **没有等待/排队的请求**

如果不满足空闲条件，API 将快速失败（HTTP 400）并且**不会修改**当前服务状态。

---

## 1. Background and implementation overview

### 1.1 Architecture / control path

The control path is:

1. **HTTP Server** (`python/sglang/srt/entrypoints/http_server.py`)
   - Exposes `PUT /hicache/storage-backend`, `DELETE /hicache/storage-backend`, `GET /hicache/storage-backend`
2. **TokenizerManager** (`python/sglang/srt/managers/tokenizer_communicator_mixin.py`)
   - Sends the request to the Scheduler via `_Communicator`
3. **Scheduler** (`python/sglang/srt/managers/scheduler.py`)
   - Performs a **strict idle check**
   - Calls `tree_cache.attach_storage_backend(...)` / `detach_storage_backend(...)`
4. **HiRadixCache** (`python/sglang/srt/mem_cache/hiradix_cache.py`)
   - Parses `hicache_storage_backend_extra_config_json` (supports both backend config and prefetch knobs)
   - Calls `cache_controller.attach_storage_backend(...)` / `detach_storage_backend(...)`
5. **HiCacheController** (`python/sglang/srt/managers/cache_controller.py`)
   - Creates/destroys the storage backend instance (via `StorageBackendFactory`)
   - Starts/stops backend background threads at runtime (prefetch/backup)

**中文对照**：## 1. 背景和实现概述

### 1.1 架构/控制路径

控制路径是：

1. **HTTP 服务器** (`python/sglang/srt/entrypoints/http_server.py`)
   - 暴露 `PUT /hicache/storage-backend`、`DELETE /hicache/storage-backend`、`GET /hicache/storage-backend`
2. **TokenizerManager** (`python/sglang/srt/managers/tokenizer_communicator_mixin.py`)
   - 通过 `_Communicator` 将请求发送到调度器
3. **调度器** (`python/sglang/srt/managers/scheduler.py`)
   - 执行**严格的空闲检查**
   - 调用 `tree_cache.attach_storage_backend(...)` / `detach_storage_backend(...)`
4. **HiRadixCache** (`python/sglang/srt/mem_cache/hiradix_cache.py`)
   - 解析 `hicache_storage_backend_extra_config_json`（支持后端配置和预取旋钮）
   - 调用 `cache_controller.attach_storage_backend(...)` / `detach_storage_backend(...)`
5. **HiCacheController** (`python/sglang/srt/managers/cache_controller.py`)
   - 创建/销毁存储后端实例（通过 `StorageBackendFactory`）
   - 在运行时启动/停止后端后台线程（预取/备份）

---

## 2. Idle-state requirement (strict)

The Scheduler uses a stricter `_is_idle_for_hicache_storage_op()`:

- `_is_no_request()` is true (covers running/overlap/pp/disagg and other active states)
- `waiting_queue` is empty
- `grammar_queue` is empty (if the grammar backend is enabled)

If the condition is not met, attach/detach returns an error like:

- `Reject attach: scheduler is not idle. #queue-req=... #running-req=...`

> Tip: before switching, drain upstream traffic and wait for the server to become idle, then call attach/detach.

**中文对照**：## 2. 空闲状态要求（严格）

调度器使用更严格的 `_is_idle_for_hicache_storage_op()`：

- `_is_no_request()` 为 true（涵盖运行/重叠/pp/分离和其他活动状态）
- `waiting_queue` 为空
- `grammar_queue` 为空（如果语法后端已启用）

如果不满足条件，attach/detach 会返回类似以下错误：

- `Reject attach: scheduler is not idle. #queue-req=... #running-req=...`

> 提示：在切换之前，排空上游流量并等待服务器变为空闲，然后调用 attach/detach。

### 2.1 DP (data parallel) semantics

When `dp_size > 1`, the tokenizer dispatches the request to **all DP scheduler instances** and aggregates their responses:

- The final `success` is **true only if all DP ranks return success**
- The final `message` concatenates messages from all DP ranks

This is intended to prevent "silent partial success", but it also means you may see:

- Overall **failure** even though **some ranks already succeeded**

Currently there is **no automatic partial rollback** across DP ranks (see TODO in code). Operationally:

- Prefer to keep backend config identical across ranks
- If attach fails, immediately call detach (best-effort/idempotent), fix config, then retry attach

**中文对照**：### 2.1 DP（数据并行）语义

当 `dp_size > 1` 时，分词器将请求分派到**所有 DP 调度器实例**并聚合它们的响应：

- 最终的 `success` **仅在所有 DP rank 都返回成功时为 true**
- 最终的 `message` 连接来自所有 DP rank 的消息

这是为了防止"静默部分成功"，但这也意味着您可能会看到：

- **尽管某些 rank 已经成功**，但整体**失败**

目前**没有跨 DP rank 的自动部分回滚**（参见代码中的 TODO）。在操作上：

- 最好保持跨 rank 的后端配置相同
- 如果 attach 失败，立即调用 detach（尽力/幂等），修复配置，然后重试 attach

---

## 3. How to use (HTTP Admin API)

The examples below assume your SGLang HTTP server is at `http://127.0.0.1:30000`.

### 3.1 Query current storage backend status

```bash
curl -s http://127.0.0.1:30000/hicache/storage-backend
```

Example response:

```json
{
  "hicache_storage_backend": "mooncake",
  "hicache_storage_backend_extra_config": "{\"master_server_address\":\"127.0.0.1:50051\", ...}"
}
```

**中文对照**：## 3. 使用方法（HTTP 管理 API）

以下示例假设您的 SGLang HTTP 服务器位于 `http://127.0.0.1:30000`。

### 3.1 查询当前存储后端状态

```bash
curl -s http://127.0.0.1:30000/hicache/storage-backend
```

示例响应：

```json
{
  "hicache_storage_backend": "mooncake",
  "hicache_storage_backend_extra_config": "{\"master_server_address\":\"127.0.0.1:50051\", ...}"
}
```

### 3.2 Attach (enable) a storage backend
```bash
curl -s -X PUT http://127.0.0.1:30000/hicache/storage-backend \
  -H 'Content-Type: application/json' \
  -d '{
    "hicache_storage_backend": "mooncake"
  }'
```

```bash
curl -s -X PUT http://127.0.0.1:30000/hicache/storage-backend \
  -H 'Content-Type: application/json' \
  -d '{
    "hicache_storage_backend": "mooncake",
    "hicache_storage_backend_extra_config_json": "{\"master_server_address\":\"127.0.0.1:50051\",\"protocol\":\"tcp\",\"global_segment_size\":\"4gb\",\"prefetch_threshold\":256}",
    "hicache_storage_prefetch_policy": "timeout"
  }'
```

Notes:

- `hicache_storage_backend_extra_config_json` can include both:
  - **Backend configuration** (e.g., Mooncake master/metadata/protocol, etc.)
  - **Prefetch configuration** (`prefetch_threshold`, `prefetch_timeout_base`, `prefetch_timeout_per_ki_token`, `hicache_storage_pass_prefix_keys`)

**中文对照**：注意：

- `hicache_storage_backend_extra_config_json` 可以同时包含：
  - **后端配置**（例如 Mooncake master/metadata/protocol 等）
  - **预取配置**（`prefetch_threshold`、`prefetch_timeout_base`、`prefetch_timeout_per_ki_token`、`hicache_storage_pass_prefix_keys`）

### 3.3 Detach (disable) the storage backend

```bash
curl -s -X DELETE http://127.0.0.1:30000/hicache/storage-backend
```

Notes:

- Detach only makes SGLang **stop using** the L3 storage backend and stops prefetch/backup threads
- It **does not automatically delete** data stored in Mooncake/HF3FS (or other remote backends)

**中文对照**：### 3.3 分离（禁用）存储后端

```bash
curl -s -X DELETE http://127.0.0.1:30000/hicache/storage-backend
```

注意：

- Detach 只是让 SGLang **停止使用** L3 存储后端并停止预取/备份线程
- 它**不会自动删除**存储在 Mooncake/HF3FS（或其它远程后端）中的数据

---

## 4. Behavior and caveats

- **No restart required**: attach/detach switches in-process at runtime
- **Must be idle**: otherwise the request is rejected to avoid consistency issues
- **Host KV layout constraints still apply**: for example, Mooncake still requires layouts like `page_first/page_first_direct/page_head`; if the server's HiCache host-memory layout does not satisfy the backend requirements, attach will fail with an error
- **Observability**:
  - After attach, `server_args.hicache_storage_backend*` is updated on both the tokenizer and scheduler sides
  - If metrics are enabled, attach will create a storage metrics collector in `HiRadixCache` on demand

**中文对照**：## 4. 行为和注意事项

- **无需重启**：attach/detach 在运行时在进程内切换
- **必须空闲**：否则请求将被拒绝以避免一致性问题
- **主机 KV 布局约束仍然适用**：例如，Mooncake 仍然需要像 `page_first/page_first_direct/page_head` 这样的布局；如果服务器的 HiCache 主机内存布局不满足后端要求，attach 将失败并报错
- **可观测性**：
  - 附加后，`server_args.hicache_storage_backend*` 在分词器和调度器两端都会更新
  - 如果启用了指标，attach 将在需要时在 `HiRadixCache` 中创建存储指标收集器

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/entrypoints/http_server.py` | 暴露 `PUT/DELETE/GET /hicache/storage-backend` HTTP 管理端点 |
| `python/sglang/srt/managers/tokenizer_communicator_mixin.py` | 通过 `_Communicator` 将 attach/detach 请求从 HTTP 转发到 Scheduler |
| `python/sglang/srt/managers/scheduler.py` | 执行严格的空闲检查（`_is_idle_for_hicache_storage_op()`），委托给 `tree_cache` |
| `python/sglang/srt/mem_cache/hiradix_cache.py` | `HiRadixCache.attach_storage_backend()` / `detach_storage_backend()` — 解析配置 JSON，调用缓存控制器 |
| `python/sglang/srt/managers/cache_controller.py` | `HiCacheController` 通过 `StorageBackendFactory` 创建/销毁存储后端实例，管理预取/备份线程 |

### 架构

```
[HTTP Admin API]
  PUT/DELETE /hicache/storage-backend
        │
        ▼
[TokenizerManager] ──_Communicator──▶ [Scheduler]
                                         │
                                    idle check:
                                    _is_no_request() &&
                                    waiting_queue empty &&
                                    grammar_queue empty
                                         │
                                         ▼
                                   [HiRadixCache]
                                    attach/detach_storage_backend()
                                         │
                                         ▼
                                   [HiCacheController]
                                    StorageBackendFactory.create()
                                    start/stop prefetch & backup threads
```

### 关键代码逻辑

- **空闲检查**: `scheduler.py` 使用 `_is_idle_for_hicache_storage_op()` 结合 `_is_no_request()`、空 `waiting_queue` 和空 `grammar_queue`
- **DP 语义**: 当 `dp_size > 1` 时，tokenizer 分发到所有 DP scheduler 并要求所有 rank 都成功（无部分回滚）
- **存储后端**: 通过 `cache_controller.py` 中的 `StorageBackendFactory` 支持 `mooncake`、`hf3fs`、`nixl`、`file`、`aibrix`、`eic`

### 集成要点

- **HTTP 端点**: `PUT /hicache/storage-backend` (attach)、`DELETE /hicache/storage-backend` (detach)、`GET /hicache/storage-backend` (查询状态)
- **配置 JSON**: `hicache_storage_backend_extra_config_json` 支持后端配置和预取参数（`prefetch_threshold`、`prefetch_timeout_base` 等）
- **运行时更新**: 成功 attach 后，`server_args.hicache_storage_backend*` 在 tokenizer 和 scheduler 两侧都会更新
