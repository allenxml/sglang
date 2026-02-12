# Production Metrics

SGLang exposes the following metrics via Prometheus. You can enable it by adding `--enable-metrics` when you launch the server.

**中文对照**：SGLang 通过 Prometheus 公开以下指标。您可以在启动服务器时添加 `--enable-metrics` 来启用它。

An example of the monitoring dashboard is available in [examples/monitoring/grafana.json](https://github.com/sgl-project/sglang/blob/main/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json).

**中文对照**：监控仪表板的示例可在 [examples/monitoring/grafana.json](https://github.com/sgl-project/sglang/blob/main/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json) 中找到。

Here is an example of the metrics:

**中文对照**：以下是指标的示例：

```
$ curl http://localhost:30000/metrics
# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.128902e+06
# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.557572e+06
# HELP sglang:token_usage The token usage
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.28
# HELP sglang:cache_hit_rate The cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.007507552643049313
# HELP sglang:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 2.3518979474117756e+06
sglang:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="0.06",model_name="meta-llama/Llama-3.1-8B-Instruct"} 3.0
sglang:time_to_first_token_seconds_bucket{le="0.08",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.25",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.75",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 27.0
sglang:time_to_first_token_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
sglang:time_to_first_token_seconds_bucket{le="5.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 314.0
sglang:time_to_first_token_seconds_bucket{le="7.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 941.0
sglang:time_to_first_token_seconds_bucket{le="10.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1330.0
sglang:time_to_first_token_seconds_bucket{le="15.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1970.0
sglang:time_to_first_token_seconds_bucket{le="20.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 2326.0
sglang:time_to_first_token_seconds_bucket{le="25.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 2417.0
sglang:time_to_first_token_seconds_bucket{le="30.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 2513.0
sglang:time_to_first_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11008.0
sglang:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 11008.0
# HELP sglang:e2e_request_latency_seconds Histogram of End-to-end request latency in seconds
# TYPE sglang:e2e_request_latency_seconds histogram
sglang:e2e_request_latency_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 3.116093850019932e+06
sglang:e2e_request_latency_seconds_bucket{le="0.3",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:e2e_request_latency_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="0.8",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="1.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="2.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="5.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.0
sglang:e2e_request_latency_seconds_bucket{le="10.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 10.0
sglang:e2e_request_latency_seconds_bucket{le="15.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11.0
sglang:e2e_request_latency_seconds_bucket{le="20.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 14.0
sglang:e2e_request_latency_seconds_bucket{le="30.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 247.0
sglang:e2e_request_latency_seconds_bucket{le="40.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 486.0
sglang:e2e_request_latency_seconds_bucket{le="50.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 845.0
sglang:e2e_request_latency_seconds_bucket{le="60.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1513.0
sglang:e2e_request_latency_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11228.0
sglang:e2e_request_latency_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 11228.0
# HELP sglang:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE sglang:time_per_output_token_seconds histogram
sglang:time_per_output_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 866964.5791549598
sglang:time_per_output_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 73.0
sglang:time_per_output_token_seconds_bucket{le="0.015",model_name="meta-llama/Llama-3.1-8B-Instruct"} 382.0
sglang:time_per_output_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 593.0
sglang:time_per_output_token_seconds_bucket{le="0.025",model_name="meta-llama/Llama-3.1-8B-Instruct"} 855.0
sglang:time_per_output_token_seconds_bucket{le="0.03",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1035.0
sglang:time_per_output_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1815.0
sglang:time_per_output_token_seconds_bucket{le="0.05",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11685.0
sglang:time_per_output_token_seconds_bucket{le="0.075",model_name="meta-llama/Llama-3.1-8B-Instruct"} 433413.0
sglang:time_per_output_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 4.950195e+06
sglang:time_per_output_token_seconds_bucket{le="0.15",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.039435e+06
sglang:time_per_output_token_seconds_bucket{le="0.2",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.171662e+06
sglang:time_per_output_token_seconds_bucket{le="0.3",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.266055e+06
sglang:time_per_output_token_seconds_bucket{le="0.4",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.296752e+06
sglang:time_per_output_token_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.312226e+06
sglang:time_per_output_token_seconds_bucket{le="0.75",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.339675e+06
sglang:time_per_output_token_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.357747e+06
sglang:time_per_output_token_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.389414e+06
sglang:time_per_output_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
sglang:time_per_output_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
# HELP sglang:func_latency_seconds Function latency in seconds
# TYPE sglang:func_latency_seconds histogram
sglang:func_latency_seconds_sum{name="generate_request"} 4.514771912145079
sglang:func_latency_seconds_bucket{le="0.05",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.07500000000000001",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.1125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.16875",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.253125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.3796875",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.56953125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.8542968750000001",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="1.2814453125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="1.9221679687500002",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="2.8832519531250003",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="4.3248779296875",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="6.487316894531251",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="9.730975341796876",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="14.596463012695313",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="21.89469451904297",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="32.84204177856446",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="49.26306266784668",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="+Inf",name="generate_request"} 14007.0
sglang:func_latency_seconds_count{name="generate_request"} 14007.0
# HELP sglang:num_running_reqs The number of running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 162.0
# HELP sglang:num_used_tokens The number of used tokens
# TYPE sglang:num_used_tokens gauge
sglang:num_used_tokens{model_name="meta-llama/Llama-3.1-8B-Instruct"} 123859.0
# HELP sglang:gen_throughput The generate throughput (token/s)
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="meta-llama/Llama-3.1-8B-Instruct"} 86.50814177726902
# HELP sglang:num_queue_reqs The number of requests in the waiting queue
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 2826.0
```

## Setup Guide

This section describes how to set up the monitoring stack (Prometheus + Grafana) provided in the `examples/monitoring` directory.

**中文对照**：本节介绍如何设置 `examples/monitoring` 目录中提供的监控栈（Prometheus + Grafana）。

### Prerequisites

- Docker and Docker Compose installed
- SGLang server running with metrics enabled

**中文对照**：
### 前置条件

- 安装 Docker 和 Docker Compose
- SGLang 服务器运行并启用了指标

### Usage

1.  **Start your SGLang server with metrics enabled:**

    ```bash
    python -m sglang.launch_server \
      --model-path <your_model_path> \
      --port 30000 \
      --enable-metrics
    ```
    Replace `<your_model_path>` with the actual path to your model (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`). Ensure the server is accessible from the monitoring stack (you might need `--host 0.0.0.0` if running in Docker). By default, the metrics endpoint will be available at `http://<sglang_server_host>:30000/metrics`.

**中文对照**：
1.  **启动启用指标的 SGLang 服务器：**

    ```bash
    python -m sglang.launch_server \
      --model-path <your_model路径> \
      --port 30000 \
      --enable-metrics
    ```
    将 `<your_model路径>` 替换为您的模型的实际路径（例如 `meta-llama/Meta-Llama-3.1-8B-Instruct`）。确保监控栈可以访问服务器（如果在 Docker 中运行，可能需要 `--host 0.0.0.0`）。默认情况下，指标端点将在 `http://<sglang服务器主机>:30000/metrics` 提供。

2.  **Navigate to the monitoring example directory:**
    ```bash
    cd examples/monitoring
    ```

**中文对照**：
2.  **导航到监控示例目录：**
    ```bash
    cd examples/monitoring
    ```

3.  **Start the monitoring stack:**
    ```bash
    docker compose up -d
    ```
    This command will start Prometheus and Grafana in the background.

**中文对照**：
3.  **启动监控栈：**
    ```bash
    docker compose up -d
    ```
    此命令将在后台启动 Prometheus 和 Grafana。

4.  **Access the monitoring interfaces:**
    *   **Grafana:** Open your web browser and go to [http://localhost:3000](http://localhost:3000).
    *   **Prometheus:** Open your web browser and go to [http://localhost:9090](http://localhost:9090).

**中文对照**：
4.  **访问监控界面：**
    *   **Grafana：** 打开您的网页浏览器并访问 [http://localhost:3000](http://localhost:3000)。
    *   **Prometheus：** 打开您的网页浏览器并访问 [http://localhost:9090](http://localhost:9090)。

5.  **Log in to Grafana:**
    *   Default Username: `admin`
    *   Default Password: `admin`
    You will be prompted to change the password upon your first login.

**中文对照**：
5.  **登录 Grafana：**
    *   默认用户名：`admin`
    *   默认密码：`admin`
    您将在首次登录时提示更改密码。

6.  **View the Dashboard:**
    The SGLang dashboard is pre-configured and should be available automatically. Navigate to `Dashboards` -> `Browse` -> `SGLang Monitoring` folder -> `SGLang Dashboard`.

**中文对照**：
6.  **查看仪表板：**
    SGLang 仪表板已预配置，应可自动使用。导航到 `Dashboards` -> `Browse` -> `SGLang Monitoring` 文件夹 -> `SGLang Dashboard`。

### Troubleshooting

*   **Port Conflicts:** If you encounter errors like "port is already allocated," check if other services (including previous instances of Prometheus/Grafana) are using ports `9090` or `3000`. Use `docker ps` to find running containers and `docker stop <container_id>` to stop them, or use `lsof -i :<port>` to find other processes using the ports. You might need to adjust the ports in the `docker-compose.yaml` file if they permanently conflict with other essential services on your system.

**中文对照**：
*   **端口冲突：** 如果遇到"端口已被分配"等错误，请检查其他服务（包括 Prometheus/Grafana 的先前实例）是否正在使用端口 `9090` 或 `3000`。使用 `docker ps` 查找正在运行的容器，并使用 `docker stop <container_id>` 停止它们，或使用 `lsof -i :<port>` 查找使用端口的其他进程。如果它们与系统上的其他必要服务永久冲突，您可能需要在 `docker-compose.yaml` 文件中调整端口。

To modify Grafana's port to the other one(like 3090) in your Docker Compose file, you need to explicitly specify the port mapping under the grafana service.

**中文对照**：要在 Docker Compose 文件中将 Grafana 的端口修改为其他端口（如 3090），需要在 grafana 服务下明确指定端口映射。

    Option 1: Add GF_SERVER_HTTP_PORT to the environment section:
    ```
      environment:
    - GF_AUTH_ANONYMOUS_ENABLED=true
    - GF_SERVER_HTTP_PORT=3090  # <-- Add this line
    ```
    Option 2: Use port mapping:
    ```
    grafana:
      image: grafana/grafana:latest
      container_name: grafana
      ports:
      - "3090:3000"  # <-- Host:Container port mapping
    ```
*   **Connection Issues:**
    *   Ensure both Prometheus and Grafana containers are running (`docker ps`).
    *   Verify the Prometheus data source configuration in Grafana (usually auto-configured via `grafana/datasources/datasource.yaml`). Go to `Connections` -> `Data sources` -> `Prometheus`. The URL should point to the Prometheus service (e.g., `http://prometheus:9090`).
    *   Confirm that your SGLang server is running and the metrics endpoint (`http://<sglang_server_host>:30000/metrics`) is accessible *from the Prometheus container*. If SGLang is running on your host machine and Prometheus is in Docker, use `host.docker.internal` (on Docker Desktop) or your machine's network IP instead of `localhost` in the `prometheus.yaml` scrape configuration.

**中文对照**：
*   **连接问题：**
    *   确保 Prometheus 和 Grafana 容器正在运行（`docker ps`）。
    *   验证 Grafana 中的 Prometheus 数据源配置（通常通过 `grafana/datasources/datasource.yaml` 自动配置）。转到 `Connections` -> `Data sources` -> `Prometheus`。URL 应指向 Prometheus 服务（例如 `http://prometheus:9090`）。
    *   确认您的 SGLang 服务器正在运行，且指标端点（`http://<sglang服务器主机>:30000/metrics`）可从 Prometheus 容器访问。如果 SGLang 在您的主机上运行而 Prometheus 在 Docker 中，请在 `prometheus.yaml` 抓取配置中使用 `host.docker.internal`（在 Docker Desktop 上）或您机器的网络 IP，而不是 `localhost`。

*   **No Data on Dashboard:**
    *   Generate some traffic to your SGLang server to produce metrics. For example, run a benchmark:
        ```bash
        python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 100 --random-input 128 --random-output 128
        ```
    *   Check the Prometheus UI (`http://localhost:9090`) under `Status` -> `Targets` to see if the SGLang endpoint is being scraped successfully.
    *   Verify the `model_name` and `instance` labels in your Prometheus metrics match the variables used in the Grafana dashboard. You might need to adjust the Grafana dashboard variables or the labels in your Prometheus configuration.

**中文对照**：
*   **仪表板无数据：**
    *   为您的 SGLang 服务器生成一些流量以产生指标。例如，运行基准测试：
        ```bash
        python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 100 --random-input 128 --random-output 128
        ```
    *   在 Prometheus UI（`http://localhost:9090`）中的 `Status` -> `Targets` 下检查，看 SGLang 端点是否被成功抓取。
    *   验证 Prometheus 指标中的 `model_name` 和 `instance` 标签与 Grafana 仪表板中使用的变量匹配。您可能需要调整 Grafana 仪表板变量或 Prometheus 配置中的标签。

### Configuration Files

The monitoring setup is defined by the following files within the `examples/monitoring` directory:

*   `docker-compose.yaml`: Defines the Prometheus and Grafana services.
*   `prometheus.yaml`: Prometheus configuration, including scrape targets.
*   `grafana/datasources/datasource.yaml`: Configures the Prometheus data source for Grafana.
*   `grafana/dashboards/config/dashboard.yaml`: Tells Grafana to load dashboards from the specified path.
*   `grafana/dashboards/json/sglang-dashboard.json`: The actual Grafana dashboard definition in JSON format.

You can customize the setup by modifying these files. For instance, you might need to update the `static_configs` target in `prometheus.yaml` if your SGLang server runs on a different host or port.

**中文对照**：监控设置由 `examples/monitoring` 目录中的以下文件定义：

*   `docker-compose.yaml`：定义 Prometheus 和 Grafana 服务。
*   `prometheus.yaml`：Prometheus 配置，包括抓取目标。
*   `grafana/datasources/datasource.yaml`：为 Grafana 配置 Prometheus 数据源。
*   `grafana/dashboards/config/dashboard.yaml`：告诉 Grafana 从指定路径加载仪表板。
*   `grafana/dashboards/json/sglang-dashboard.json`：实际的 Grafana 仪表板 JSON 格式定义。

您可以通过修改这些文件来自定义设置。例如，如果您的 SGLang 服务器在不同的主机或端口上运行，您可能需要更新 `prometheus.yaml` 中的 `static_configs` 目标。

#### Check if the metrics are being collected

Run:
```
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 3000 \
  --random-input 1024 \
  --random-output 1024 \
  --random-range-ratio 0.5
```

to generate some requests.

**中文对照**：#### 检查指标是否正在收集

运行：
```
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 3000 \
  --random-input 1024 \
  --random-output 1024 \
  --random-range-ratio 0.5
```
以生成一些请求。

Then you should be able to see the metrics in the Grafana dashboard.

**中文对照**：然后您应该能够在 Grafana 仪表板中看到指标。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/managers/scheduler_metrics_mixin.py` | Prometheus 指标定义：所有 `sglang:*` 指标的计数器、仪表、直方图 |
| `python/sglang/srt/managers/request_metrics_exporter.py` | 每请求指标导出器：启用 `--export-metrics-to-file` 时将详细指标写入文件 |
| `python/sglang/srt/entrypoints/http_server.py` | `/metrics` 端点：通过 HTTP 公开 Prometheus 格式的指标 |
| `examples/monitoring/` | Grafana + Prometheus 监控栈：docker-compose、仪表板、数据源配置 |

### 关键代码逻辑

- **指标类型**：`prompt_tokens_total`（计数器）、`token_usage`（仪表）、`time_to_first_token_seconds`（直方图）、`gen_throughput`（仪表）
- **指标注册**：`scheduler_metrics_mixin.py` 在调度器初始化时创建 Prometheus 收集器
- **指标更新**：调度器在每次前向迭代后更新指标（运行请求、队列深度、吞吐量）
- **仪表板**：`examples/monitoring/grafana/dashboards/json/sglang-dashboard.json` 提供预构建的 Grafana 面板

### 集成要点

- **启用**：`--enable-metrics` 标志激活 Prometheus 指标收集和 `/metrics` 端点
- **多调度器**：`--enable-metrics-for-all-schedulers` 在 DP 设置中公开每调度器指标
- **自定义标签**：`--tokenizer-metrics-custom-labels-header` 允许通过 HTTP 头注入自定义标签
- **监控栈**：`docker compose -f examples/monitoring/docker-compose.yaml up -d` 启动 Prometheus + Grafana
