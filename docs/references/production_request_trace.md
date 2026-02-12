# Production Request Tracing

SGLang exports request trace data based on the OpenTelemetry Collector. You can enable tracing by adding the `--enable-trace` and configure the OpenTelemetry Collector endpoint using `--otlp-traces-endpoint` when launching the server.

**中文对照**：SGLang 基于 OpenTelemetry Collector 导出请求跟踪数据。您可以通过添加 `--enable-trace` 启用跟踪，并在启动服务器时使用 `--otlp-traces-endpoint` 配置 OpenTelemetry Collector 端点。

You can find example screenshots of the visualization in https://github.com/sgl-project/sglang/issues/8965.

**中文对照**：您可以在 https://github.com/sgl-project/sglang/issues/8965 中找到可视化的示例截图。

## Setup Guide
This section explains how to configure the request tracing and export the trace data.
1. Install the required packages and tools
    * install Docker and Docker Compose
    * install the dependencies
    ```bash
    # enter the SGLang root directory
    pip install -e "python[tracing]"

    # or manually install the dependencies using pip
    pip install opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp opentelemetry-exporter-otlp-proto-grpc
    ```

**中文对照**：
## 设置指南
本节介绍如何配置请求跟踪和导出跟踪数据。
1. 安装所需的包和工具
    * 安装 Docker 和 Docker Compose
    * 安装依赖项
    ```bash
    # 进入 SGLang 根目录
    pip install -e "python[tracing]"

    # 或使用 pip 手动安装依赖项
    pip install opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp opentelemetry-exporter-otlp-proto-grpc
    ```

2. launch opentelemetry collector and jaeger
    ```bash
    docker compose -f examples/monitoring/tracing_compose.yaml up -d
    ```

**中文对照**：
2. 启动 opentelemetry collector 和 jaeger
    ```bash
    docker compose -f examples/monitoring/tracing_compose.yaml up -d
    ```

3. start your SGLang server with tracing enabled
    ```bash
    # set env variables
    export SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500
    export SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=64
    # start the prefill and decode server
    python -m sglang.launch_server --enable-trace --otlp-traces-endpoint 0.0.0.0:4317 <other option>
    # start the mini lb
    python -m sglang_router.launch_router --enable-trace --otlp-traces-endpoint 0.0.0.0:4317 <other option>
    ```

    Replace `0.0.0.0:4317` with the actual endpoint of the opentelemetry collector. If you launched the openTelemetry collector with tracing_compose.yaml, the default receiving port is 4317.

**中文对照**：
3. 启动启用跟踪的 SGLang 服务器
    ```bash
    # 设置环境变量
    export SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500
    export SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=64
    # 启动 prefill 和 decode 服务器
    python -m sglang.launch_server --enable-trace --otlp-traces-endpoint 0.0.0.0:4317 <其他选项>
    # 启动 mini lb
    python -m sglang_router.launch_router --enable-trace --otlp-traces-endpoint 0.0.0.0:4317 <其他选项>
    ```

    将 `0.0.0.0:4317` 替换为 opentelemetry collector 的实际端点。如果您使用 tracing_compose.yaml 启动了 openTelemetry collector，则默认接收端口为 4317。

    To use the HTTP/protobuf span exporter, set the following environment variable and point to an HTTP endpoint, for example, `http://0.0.0.0:4318/v1/traces`.
    ```bash
    export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
    ```

**中文对照**：
    要使用 HTTP/protobuf span exporter，请设置以下环境变量并指向 HTTP 端点，例如 `http://0.0.0.0:4318/v1/traces`。
    ```bash
    export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
    ```

4. raise some requests
5. Observe whether trace data is being exported
    * Access port 16686 of Jaeger using a web browser to visualize the request traces.
    * The OpenTelemetry Collector also exports trace data in JSON format to /tmp/otel_trace.json. In a follow-up patch, we will provide a tool to convert this data into a Perfetto-compatible format, enabling visualization of requests in the Perfetto UI.

**中文对照**：
4. 发起一些请求
5. 观察跟踪数据是否正在导出
    * 使用网页浏览器访问 Jaeger 的 16686 端口以可视化请求跟踪。
    * OpenTelemetry Collector 还将跟踪数据以 JSON 格式导出到 /tmp/otel_trace.json。在后续更新中，我们将提供一种工具将此数据转换为 Perfetto 兼容格式，从而在 Perfetto UI 中可视化请求。

## How to add Tracing for slices you're interested in?
We have already inserted instrumentation points in the tokenizer and scheduler main threads. If you wish to trace additional request execution segments or perform finer-grained tracing, please use the APIs from the tracing package as described below.

**中文对照**：
## 如何为您感兴趣的部分添加跟踪？
我们已经在分词器和调度器主线程中插入了检测点。如果您希望跟踪其他请求执行段或执行更细粒度的跟踪，请按照以下描述使用跟踪包中的 API。

1. initialization

    Every process involved in tracing during the initialization phase should execute:
    ```python
    process_tracing_init(otlp_traces_endpoint, server_name)
    ```
    The otlp_traces_endpoint is obtained from the arguments, and you can set server_name freely, but it should remain consistent across all processes.

**中文对照**：
1. 初始化

    在初始化阶段参与跟踪的每个进程应执行：
    ```python
    process_tracing_init(otlp_traces_endpoint, server_name)
    ```
    otlp_traces_endpoint 从参数获取，您可以自由设置 server_name，但它应在所有进程之间保持一致。

    Every thread involved in tracing during the initialization phase should execute:
    ```python
    trace_set_thread_info("thread label", tp_rank, dp_rank)
    ```
    The "thread label" can be regarded as the name of the thread, used to distinguish different threads in the visualization view.

**中文对照**：
    在初始化阶段参与跟踪的每个线程应执行：
    ```python
    trace_set_thread_info("thread label", tp_rank, dp_rank)
    ```
    "thread label" 可以被视为线程的名称，用于在可视化视图中区分不同的线程。

2. Mark the beginning and end of a request
    ```
    trace_req_start(rid, bootstrap_room)
    trace_req_finish(rid)
    ```
    These two APIs must be called within the same process, for example, in the tokenizer.

**中文对照**：
这两个 API 必须在同一个进程中调用，例如在分词器中。

3. Add tracing for slice

    * Add slice tracing normally:
        ```python
        trace_slice_start("slice A", rid)
        trace_slice_end("slice A", rid)
        ```

    - Use the "anonymous" flag to not specify a slice name at the start of the slice, allowing the slice name to be determined by trace_slice_end.
    <br>Note: Anonymous slices must not be nested.
        ```python
        trace_slice_start("", rid, anonymous = True)
        trace_slice_end("slice A", rid)
        ```

**中文对照**：
    - 使用"anonymous"标志不在切片开始时指定切片名称，允许切片名称由 trace_slice_end 确定。
    <br>注意：匿名切片不能嵌套。
        ```python
        trace_slice_start("", rid, anonymous = True)
        trace_slice_end("slice A", rid)
        ```

    - In trace_slice_end, use auto_next_anon to automatically create the next anonymous slice, which can reduce the number of instrumentation points needed.
        ```python
        trace_slice_start("", rid, anonymous = True)
        trace_slice_end("slice A", rid, auto_next_anon = True)
        trace_slice_end("slice B", rid, auto_next_anon = True)
        trace_slice_end("slice C", rid, auto_next_anon = True)
        trace_slice_end("slice D", rid)
        ```

**中文对照**：
    - 在 trace_slice_end 中，使用 auto_next_anon 自动创建下一个匿名切片，这可以减少所需的检测点数量。
        ```python
        trace_slice_start("", rid, anonymous = True)
        trace_slice_end("slice A", rid, auto_next_anon = True)
        trace_slice_end("slice B", rid, auto_next_anon = True)
        trace_slice_end("slice C", rid, auto_next_anon = True)
        trace_slice_end("slice D", rid)
        ```

    - The end of the last slice in a thread must be marked with thread_finish_flag=True; otherwise, the thread's span will not be properly generated.
        ```python
        trace_slice_end("slice D", rid, thread_finish_flag = True)
        ```

**中文对照**：
    - 线程中最后一个切片的结束必须用 thread_finish_flag=True 标记；否则，线程的 span 将无法正确生成。
        ```python
        trace_slice_end("slice D", rid, thread_finish_flag = True)
        ```

4. When the request execution flow transfers to another thread, the trace context needs to be explicitly propagated.
    - sender: Execute the following code before sending the request to another thread via ZMQ
        ```python
        trace_context = trace_get_proc_propagate_context(rid)
        req.trace_context = trace_context
        ```
    - receiver: Execute the following code after receiving the request via ZMQ
        ```python
        trace_set_proc_propagate_context(rid, req.trace_context)
        ```

**中文对照**：
4. 当请求执行流程转移到另一个线程时，需要显式传播跟踪上下文。
    - 发送者：通过 ZMQ 将请求发送到另一个线程之前执行以下代码
        ```python
        trace_context = trace_get_proc_propagate_context(rid)
        req.trace_context = trace_context
        ```
    - 接收者：通过 ZMQ 收到请求后执行以下代码
        ```python
        trace_set_proc_propagate_context(rid, req.trace_context)
        ```

5. When the request execution flow transfers to another node(PD disaggregation), the trace context needs to be explicitly propagated.
    - sender: Execute the following code before sending the request to node thread via http
        ```python
        trace_context = trace_get_remote_propagate_context(bootstrap_room_list)
        headers = {"trace_context": trace_context}
        session.post(url, headers=headers)
        ```
    - receiver: Execute the following code after receiving the request via http
        ```python
        trace_set_remote_propagate_context(request.headers['trace_context'])
        ```

**中文对照**：
5. 当请求执行流程转移到另一个节点（PD 分离）时，需要显式传播跟踪上下文。
    - 发送者：通过 http 将请求发送到节点线程之前执行以下代码
        ```python
        trace_context = trace_get_remote_propagate_context(bootstrap_room_list)
        headers = {"trace_context": trace_context}
        session.post(url, headers=headers)
        ```
    - 接收者：通过 http 收到请求后执行以下代码
        ```python
        trace_set_remote_propagate_context(request.headers['trace_context'])
        ```

## How to Extend the Tracing Framework to Support Complex Tracing Scenarios

The currently provided tracing package still has potential for further development. If you wish to build more advanced features upon it, you must first understand its existing design principles.

**中文对照**：
## 如何扩展跟踪框架以支持复杂的跟踪场景

当前提供的跟踪包仍有进一步开发的潜力。如果您希望在其上构建更高级的功能，必须首先了解其现有的设计原则。

The core of the tracing framework's implementation lies in the design of the span structure and the trace context. To aggregate scattered slices and enable concurrent tracking of multiple requests, we have designed a two-level trace context structure and a four-level span structure: `SglangTraceReqContext`, `SglangTraceThreadContext`. Their relationship is as follows:
```
SglangTraceReqContext (req_id="req-123")
├── SglangTraceThreadContext(thread_label="scheduler", tp_rank=0)
|
└── SglangTraceThreadContext(thread_label="scheduler", tp_rank=1)
```

Each traced request maintains a global `SglangTraceReqContext`. For every thread processing the request, a corresponding `SglangTraceThreadContext` is recorded and composed within the `SglangTraceReqContext`. Within each thread, every currently traced slice (possibly nested) is stored in a list.

**中文对照**：跟踪框架实现的核心在于 span 结构和跟踪上下文的设计。为了聚合分散的切片并启用多个请求的并发跟踪，我们设计了两级跟踪上下文结构和四级 span 结构：`SglangTraceReqContext`、`SglangTraceThreadContext`。它们的关系如下：
```
SglangTraceReqContext (req_id="req-123")
├── SglangTraceThreadContext(thread_label="scheduler", tp_rank=0)
|
└── SglangTraceThreadContext(thread_label="scheduler", tp_rank=1)
```

每个跟踪的请求维护一个全局的 `SglangTraceReqContext`。对于处理请求的每个线程，相应的 `SglangTraceThreadContext` 被记录并在 `SglangTraceReqContext` 内组合。在每个线程内，当前每个被跟踪的切片（可能是嵌套的）存储在一个列表中。

In addition to the above hierarchy, each slice also records its previous slice via Span.add_link(), which can be used to trace the execution flow.

**中文对照**：除了上述层次结构外，每个切片还通过 Span.add_link() 记录其前一个切片，可用于跟踪执行流程。

When the request execution flow transfers to a new thread, the trace context needs to be explicitly propagated. In the framework, this is represented by `SglangTracePropagateContext`, which contains the context of the request span and the previous slice span.

**中文对照**：当请求执行流程转移到新线程时，需要显式传播跟踪上下文。在框架中，这由 `SglangTracePropagateContext` 表示，它包含请求 span 和前一个切片上下文的上下文。


We designed a four-level span structure, consisting of `bootstrap_room_span`, `req_root_span`, `thread_span`, and `slice_span`. Among them, `req_root_span` and `thread_span` correspond to `SglangTraceReqContext` and `SglangTraceThreadContext`, respectively, and `slice_span` is stored within the `SglangTraceThreadContext`. The `bootstrap_room_span` is designed to accommodate the separation of PD-disaggregation. On different nodes, we may want to add certain attributes to the `req_root_span`. However, if the `req_root_span` is shared across all nodes, the Prefill and Decode nodes would not be allowed to add attributes due to the constraints imposed by OpenTelemetry's design.

**中文对照**：我们设计了四级 span 结构，由 `bootstrap_room_span`、`req_root_span`、`thread_span` 和 `slice_span` 组成。其中，`req_root_span` 和 `thread_span` 分别对应 `SglangTraceReqContext` 和 `SglangTraceThreadContext`，而 `slice_span` 存储在 `SglangTraceThreadContext` 内。`bootstrap_room_span` 旨在适应 PD 分离的分离。在不同节点上，我们可能希望向 `req_root_span` 添加某些属性。但是，如果 `req_root_span` 在所有节点之间共享，由于 OpenTelemetry 设计的约束，Prefill 和 Decode 节点将不允许添加属性。

```
bootstrap room span
├── router req root span
|    └── router thread span
|          └── slice span
├── prefill req root span
|    ├── tokenizer thread span
|    |     └── slice span
|    └── scheduler thread span
|          └── slice span
└── decode req root span
      ├── tokenizer thread span
      |    └── slice span
      └── scheduler thread span
           └── slice span
```

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/tracing/trace.py` | 链路追踪核心：`process_tracing_init()`、`trace_req_start/finish()`、`trace_slice_start/end()` 以及 span 结构管理 |
| `python/sglang/srt/managers/tokenizer_manager.py` | 追踪集成：请求的开始/结束标记，通过 ZMQ 进行跨线程上下文传播 |
| `python/sglang/srt/managers/scheduler.py` | 追踪集成：调度器线程中 prefill/decode 阶段的 slice 埋点 |
| `python/sglang/srt/server_args.py` | `--enable-trace` 和 `--otlp-traces-endpoint` 命令行参数定义 |
| `examples/monitoring/tracing_compose.yaml` | Docker Compose 配置：OpenTelemetry Collector + Jaeger 可视化服务栈 |

### 关键代码逻辑

- **Span 层级结构**：四层设计 — `bootstrap_room_span` → `req_root_span` → `thread_span` → `slice_span`
- **上下文类**：`SglangTraceReqContext`（每请求）和 `SglangTraceThreadContext`（每线程）构成两级追踪上下文
- **跨进程传播**：通过 `trace_get_proc_propagate_context()` / `trace_set_proc_propagate_context()` 在 ZMQ 消息中传递追踪上下文
- **跨节点传播**：通过 `trace_get_remote_propagate_context()` / `trace_set_remote_propagate_context()` 在 PD 分离架构的 HTTP 请求中传递
- **匿名切片**：`trace_slice_start("", rid, anonymous=True)` 配合 `auto_next_anon` 减少埋点代码量

### 集成要点

- **启用方式**：`--enable-trace --otlp-traces-endpoint <host>:<port>` 激活 OpenTelemetry span 导出
- **导出器配置**：通过环境变量 `SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS` 和 `SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE` 控制导出行为
- **可视化**：Jaeger UI 在端口 16686 查看；JSON 格式导出到 `/tmp/otel_trace.json` 可用 Perfetto 打开
- **PD 分离支持**：`bootstrap_room_span` 允许 prefill 和 decode 节点各自独立地为其 `req_root_span` 添加属性
