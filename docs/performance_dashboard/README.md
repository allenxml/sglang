# SGLang Performance Dashboard

A web-based dashboard for visualizing SGLang nightly test performance metrics.

**中文对照**：SGLang 性能仪表板

用于可视化 SGLang 夜间测试性能指标的 Web 仪表板。

## Features

- **Performance Trends**: View throughput, latency, and TTFT trends over time
- **Model Comparison**: Compare performance across different models and configurations
- **Filtering**: Filter by GPU configuration, model, variant, and batch size
- **Interactive Charts**: Zoom, pan, and hover for detailed metrics
- **Run History**: View recent benchmark runs with links to GitHub Actions

**中文对照**：功能特性

- **性能趋势**：查看吞吐量、延迟和 TTFT 随时间变化的趋势
- **模型比较**：比较不同模型和配置的性能
- **筛选**：按 GPU 配置、模型、变体和批大小进行筛选
- **交互式图表**：缩放、平移和悬停以查看详细指标
- **运行历史**：查看最近的基准运行并链接到 GitHub Actions

## Quick Start

### Option 1: Run with Local Server (Recommended)

For live data from GitHub Actions artifacts:

```bash
# Install requirements
pip install requests

# Run the server
python server.py --fetch-on-start

# Visit http://localhost:8000
```

The server provides:
- Automatic fetching of metrics from GitHub
- Caching to reduce API calls
- `/api/metrics` endpoint for the frontend

**中文对照**：快速开始

### 选项 1：使用本地服务器运行（推荐）

获取来自 GitHub Actions 工件的实时数据：

服务器提供：
- 从 GitHub 自动获取指标
- 缓存以减少 API 调用
- `/api/metrics` 端点供前端使用

### Option 2: Fetch Data Manually

Use the fetch script to download metrics data:

```bash
# Fetch last 30 days of metrics
python fetch_metrics.py --output metrics_data.json

# Fetch a specific run
python fetch_metrics.py --run-id 21338741812 --output single_run.json

# Fetch only scheduled (nightly) runs
python fetch_metrics.py --scheduled-only --days 7
```

**中文对照**：选项 2：手动获取数据

## GitHub Token

To download artifacts from GitHub, you need authentication:

1. **Using `gh` CLI** (recommended):
   ```bash
   gh auth login
   ```

2. **Using environment variable**:
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```

Without a token, the dashboard will show run metadata but not detailed benchmark results.

**中文对照**：GitHub 令牌

要从 GitHub 下载工件，您需要身份验证：

1. **使用 `gh` CLI**（推荐）：
2. **使用环境变量**：

没有令牌，仪表板将显示运行元数据但不显示详细的基准测试结果。

## Data Structure

The metrics JSON has this structure:

```json
{
  "run_id": "21338741812",
  "run_date": "2026-01-25T22:24:02.090218+00:00",
  "commit_sha": "5cdb391...",
  "branch": "main",
  "results": [
    {
      "gpu_config": "8-gpu-h200",
      "partition": 0,
      "model": "deepseek-ai/DeepSeek-V3.1",
      "variant": "TP8+MTP",
      "benchmarks": [
        {
          "batch_size": 1,
          "input_len": 4096,
          "output_len": 512,
          "latency_ms": 2400.72,
          "input_throughput": 21408.64,
          "output_throughput": 231.74,
          "overall_throughput": 1919.43,
          "ttft_ms": 191.32,
          "acc_length": 3.19
        }
      ]
    }
  ]
}
```

**中文对照**：数据结构

## Deployment

### GitHub Pages

The dashboard can be deployed to GitHub Pages for public access:

1. Copy the dashboard files to `docs/performance_dashboard/`
2. Enable GitHub Pages in repository settings
3. Set up a GitHub Action to periodically update metrics data

### Self-Hosted

For a self-hosted deployment with live data:

1. Set up a server running `server.py`
2. Configure a cron job or systemd timer to refresh data
3. Optionally put behind nginx/caddy for SSL

**中文对照**：部署

### GitHub Pages

### 自托管

## Metrics Explained

- **Overall Throughput**: Total tokens (input + output) processed per second
- **Input Throughput**: Input tokens processed per second (prefill speed)
- **Output Throughput**: Output tokens generated per second (decode speed)
- **Latency**: End-to-end time to complete the request
- **TTFT**: Time to First Token - time until the first output token
- **Acc Length**: Acceptance length for speculative decoding (MTP variants)

**中文对照**：指标说明

## Contributing

To add support for new metrics or visualizations:

1. Update `fetch_metrics.py` if data collection needs changes
2. Modify `app.js` to add new chart types or filters
3. Update `index.html` for UI changes

**中文对照**：贡献

## Troubleshooting

**No data displayed**
- Check browser console for errors
- Verify GitHub API is accessible
- Try running with `server.py --fetch-on-start`

**API rate limits**
- Use a GitHub token for higher limits
- The server caches data for 5 minutes

**Charts not rendering**
- Ensure Chart.js is loading from CDN
- Check for JavaScript errors in console

**中文对照**：故障排除

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `docs/performance_dashboard/server.py` | 本地服务器：从 GitHub Actions 获取指标数据，提供 `/api/metrics` 端点 |
| `docs/performance_dashboard/fetch_metrics.py` | 数据采集脚本：从 GitHub Actions artifacts 下载基准测试结果 |
| `docs/performance_dashboard/app.js` | 前端交互：Chart.js 图表渲染、筛选、缩放等交互逻辑 |
| `docs/performance_dashboard/index.html` | 仪表盘页面：数据展示 UI 布局 |

### 集成要点

- **数据来源**：从 GitHub Actions 的 nightly benchmark 工件中提取性能指标（吞吐量、延迟、TTFT 等）
- **认证方式**：推荐使用 `gh auth login` 或设置 `GITHUB_TOKEN` 环境变量以访问 GitHub API
- **指标说明**：Overall Throughput（总吞吐量）、TTFT（首 token 延迟）、Acc Length（推测解码的接受长度）
- **部署方式**：支持 GitHub Pages 静态部署或带数据刷新的自托管部署
