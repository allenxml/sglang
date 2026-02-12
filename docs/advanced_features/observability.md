# Observability

## Production Metrics
SGLang exposes the following metrics via Prometheus. You can enable them by adding `--enable-metrics` when launching the server.
You can query them by:
```
curl http://localhost:30000/metrics
```

See [Production Metrics](../references/production_metrics.md) and [Production Request Tracing](../references/production_request_trace.md) for more details.

## Logging

By default, SGLang does not log any request contents. You can log them by using `--log-requests`.
You can control the verbosity by using `--log-request-level`.
See [Logging](server_arguments.md#logging) for more details.

## Request Dump and Replay

You can dump all requests and replay them later for benchmarking or other purposes.

To start dumping, use the following command to send a request to a server:
```
python3 -m sglang.srt.managers.configure_logging --url http://localhost:30000 --dump-requests-folder /tmp/sglang_request_dump --dump-requests-threshold 100
```
The server will dump the requests into a pickle file for every 100 requests.

To replay the request dump, use `scripts/playground/replay_request_dump.py`.

**中文对照**：# 可观测性

## 生产指标
SGLang 通过 Prometheus 暴露以下指标。您可以在启动服务器时添加 `--enable-metrics` 来启用它们。
您可以通过以下方式查询它们：
```
curl http://localhost:30000/metrics
```

有关更多详细信息，请参阅[生产指标](../references/production_metrics.md)和[生产请求追踪](../references/production_request_trace.md)。

## 日志记录

默认情况下，SGLang 不记录任何请求内容。您可以使用 `--log-requests` 来记录它们。
您可以使用 `--log-request-level` 来控制详细程度。
有关更多详细信息，请参阅[日志记录](server_arguments.md#logging)。

## 请求转储和重放

您可以转储所有请求，以后重放它们以进行基准测试或其他目的。

要开始转储，使用以下命令向服务器发送请求：
```
python3 -m sglang.srt.managers.configure_logging --url http://localhost:30000 --dump-requests-folder /tmp/sglang_request_dump --dump-requests-threshold 100
```
服务器将每 100 个请求将请求转储到一个 pickle 文件中。

要重放请求转储，请使用 `scripts/playground/replay_request_dump.py`。

## 代码实现

### 核心文件
- `python/sglang/srt/managers/request_metrics_exporter.py`: 处理向外部系统导出指标。
- `python/sglang/srt/metrics/collector.py`: 收集和聚合指标的中央仓库。
- `python/sglang/srt/managers/scheduler_metrics_mixin.py`: 将指标收集集成到调度循环中。

### 架构
可观测性系统遵循生产者-消费者模式。各种组件（Scheduler、ModelRunner、Tokenizer）使用 `collector.py` 记录事件。`request_metrics_exporter.py` 运行一个 HTTP 服务器（通常与模型服务器在同一端口），暴露 `/metrics` 端点。当 Prometheus 抓取此端点时，导出器从收集器收集当前状态并将其格式化为 Prometheus 文本协议。

### 关键代码逻辑
在导出器中注册指标：
```python
# request_metrics_exporter.py
def init_metrics(self):
    self.request_latency = Histogram("sglang:request_latency", ...)
    self.token_throughput = Counter("sglang:token_throughput", ...)
```
在请求处理期间记录指标：
```python
# scheduler_metrics_mixin.py
def finish_request(self, request):
    self.metrics_collector.record_latency(request.latency)
    self.metrics_collector.inc_completed_requests()
```

### 集成要点
如果设置了 `--enable-metrics` 标志，可观测性系统在 `entrypoints/http_server.py` 中初始化。它通过 ZMQ 消息或在共享内存模式下的直接调用，拦截异步管道管理器的请求完成和 token 生成事件。
