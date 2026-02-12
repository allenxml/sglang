# ================================================================================
# 📊 请求指标导出器 (Request Metrics Exporter)
# ================================================================================
#
# 【这个文件是什么】What This File Does
# 这个文件定义了请求级别性能指标的导出系统，用于将每个请求的性能数据（延迟、吞吐量等）
# 持久化到外部存储（文件、数据库、Prometheus等），为生产环境监控和分析提供数据基础。
#
# 【生活比喻】Metaphor
# 想象这是一个"体检报告打印中心"：
# - 每个请求 = 一位病人
# - 性能指标 = 体检项目结果（心率、血压、血糖等）
# - RequestMetricsExporter = 打印机（可以打印到纸上、存到数据库、发送邮件）
# - RequestMetricsExporterManager = 打印中心（管理多台打印机同时工作）
#
# 【核心架构】Architecture
# 1. RequestMetricsExporter（抽象基类）
#    ├─ 定义统一的导出接口
#    └─ 提供通用的数据格式化方法
#
# 2. FileRequestMetricsExporter（文件导出实现）
#    ├─ 将指标写入本地日志文件
#    ├─ 按小时滚动日志（避免单文件过大）
#    └─ JSON Lines 格式（每行一个JSON对象）
#
# 3. RequestMetricsExporterManager（导出器管理器）
#    ├─ 支持同时启用多个导出器
#    ├─ 统一调度写入操作
#    └─ 支持私有插件扩展
#
# 【典型指标】Typical Metrics
# - 延迟类：e2e_latency_ms, prefill_latency_ms, decode_latency_ms
# - 吞吐量：input_tokens, output_tokens, throughput_tps
# - 资源：queue_wait_ms, batch_size, gpu_memory_used
#
# 【使用方式】Usage
# 启动服务时添加 --export-metrics-to-file 参数：
#   python -m sglang.launch_server \
#     --export-metrics-to-file \
#     --export-metrics-to-file-dir ./logs/metrics
#
# ================================================================================

import asyncio
import dataclasses
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Union

from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput  # 请求输入数据结构
from sglang.srt.server_args import ServerArgs  # 服务器启动参数

logger = logging.getLogger(__name__)

# ======== 需要排除的字段（不可JSON序列化）========
# Fields that should always be excluded from request parameters
# because they contain non-JSON-serializable objects (e.g., ImageData, tensors)
#
# 这些字段包含二进制数据（图像、视频、张量），无法直接转为 JSON
ALWAYS_EXCLUDE_FIELDS = {"image_data", "video_data", "audio_data", "input_embeds"}


# ======== 抽象基类：RequestMetricsExporter ========
class RequestMetricsExporter(ABC):
    """Abstract base class for exporting request-level performance metrics to a data destination."""

    def __init__(
        self,
        server_args: ServerArgs,
        obj_skip_names: Optional[set[str]],
        out_skip_names: Optional[set[str]],
    ):
        self.server_args = server_args
        self.obj_skip_names = obj_skip_names or set()
        self.out_skip_names = out_skip_names or set()

    def _format_output_data(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], out_dict: dict
    ) -> dict:
        """Format request-level output data containing performance metrics. This method
        should be called prior to writing the data record with `self.write_record()`."""

        request_params = {}
        for field in dataclasses.fields(obj):
            field_name = field.name
            # Skip fields in obj_skip_names or fields that are always excluded (not JSON serializable)
            if (
                field_name not in self.obj_skip_names
                and field_name not in ALWAYS_EXCLUDE_FIELDS
            ):
                value = getattr(obj, field_name)
                # Convert to serializable format
                if value is not None:
                    request_params[field_name] = value

        meta_info = out_dict.get("meta_info", {})
        filtered_out_meta_info = {
            k: v for k, v in meta_info.items() if k not in self.out_skip_names
        }

        request_output_data = {
            "request_parameters": json.dumps(request_params),
            **filtered_out_meta_info,
        }
        return request_output_data

    @abstractmethod
    async def write_record(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], out_dict: dict
    ):
        """Write a data record corresponding to a single request, containing performance metric data."""
        pass


# ======== 文件导出实现：FileRequestMetricsExporter ========
class FileRequestMetricsExporter(RequestMetricsExporter):
    """
    文件导出器：将请求指标写入本地日志文件 (JSON Lines 格式)

    Lightweight `RequestMetricsExporter` implementation that writes records to files on disk.

    Records are written to files in the directory specified by `--export-metrics-to-file-dir`
    server launch flag. File names are of the form `"sglang-request-metrics-{hour_suffix}.log"`.

    【核心特性】
    - 按小时滚动：每小时创建新文件（如 sglang-request-metrics-20260211_14.log）
    - JSON Lines 格式：每行一个 JSON 对象，易于流式解析
    - 异步写入：避免阻塞主线程

    【文件格式示例】
    {"request_parameters": {...}, "e2e_latency_ms": 1234, "input_tokens": 50, ...}
    {"request_parameters": {...}, "e2e_latency_ms": 2345, "input_tokens": 120, ...}
    """

    def __init__(
        self,
        server_args: ServerArgs,
        obj_skip_names: Optional[set[str]],
        out_skip_names: Optional[set[str]],
    ):
        super().__init__(server_args, obj_skip_names, out_skip_names)
        self.export_dir = getattr(server_args, "export_metrics_to_file_dir")
        os.makedirs(self.export_dir, exist_ok=True)  # 创建目录（如果不存在）

        # ======== 文件句柄状态管理 ========
        # File handler state management
        self._current_file_handler = None  # 当前打开的文件句柄
        self._current_hour_suffix = None  # 当前小时后缀（如 "20260211_14"）

    def _ensure_file_handler(self, hour_suffix: str):
        """
        确保当前小时对应的文件句柄已打开（按小时滚动日志）

        Ensure the file handler is open for the current hour suffix.

        【工作原理】
        - 如果当前小时与上次不同 → 关闭旧文件，打开新文件
        - 如果当前小时与上次相同 → 复用已打开的文件句柄
        """
        if self._current_hour_suffix != hour_suffix:
            # ======== 关闭旧的文件句柄（如果存在）========
            # Close previous file handler if it exists
            if self._current_file_handler is not None:
                try:
                    self._current_file_handler.close()
                except Exception as e:
                    logger.warning(f"Failed to close previous file handler: {e}")

            # Open new file handler
            log_filename = f"sglang-request-metrics-{hour_suffix}.log"
            log_filepath = os.path.join(self.export_dir, log_filename)

            try:
                self._current_file_handler = open(log_filepath, "a", encoding="utf-8")
                self._current_hour_suffix = hour_suffix
            except Exception as e:
                logger.error(f"Failed to open log file {log_filepath}: {e}")
                self._current_file_handler = None
                self._current_hour_suffix = None
                raise

    def close(self):
        """Close the current file handler."""
        if self._current_file_handler is not None:
            try:
                self._current_file_handler.close()
            except Exception as e:
                logger.warning(f"Failed to close file handler: {e}")
            finally:
                self._current_file_handler = None
                self._current_hour_suffix = None

    async def write_record(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], out_dict: dict
    ):
        """
        将单个请求的指标数据写入文件（异步，避免阻塞）

        【工作流程】
        1. 过滤健康检查请求（不记录）
        2. 获取当前小时后缀（如 "20260211_14"）
        3. 确保对应小时的文件已打开
        4. 格式化指标数据为 JSON
        5. 异步写入文件（使用 asyncio.to_thread 避免阻塞主线程）
        """
        # ======== 第1步：过滤健康检查请求 ========
        # Do not log health check requests, since they don't represent real user requests.
        if isinstance(obj.rid, str) and "HEALTH_CHECK" in obj.rid:
            return

        try:
            # ======== 第2步：获取当前小时后缀 ========
            # Get the log file path for the current time.
            current_time = datetime.now()
            hour_suffix = current_time.strftime("%Y%m%d_%H")  # 格式：20260211_14

            # ======== 第3步：确保对应文件已打开 ========
            # Ensure correct file handler is open for current hour
            self._ensure_file_handler(hour_suffix)

            if self._current_file_handler is None:
                return  # 文件打开失败，跳过

            # ======== 第4步：格式化指标数据 ========
            metrics_data = self._format_output_data(obj, out_dict)

            # ======== 第5步：异步写入文件 ========
            # 使用 asyncio.to_thread 在线程池中执行阻塞的文件写入操作
            def write_file():
                json.dump(metrics_data, self._current_file_handler)  # 写入JSON对象
                self._current_file_handler.write("\n")  # 换行（JSON Lines 格式）
                self._current_file_handler.flush()  # 立即刷新到磁盘

            await asyncio.to_thread(write_file)
        except Exception as e:
            logger.exception(f"Failed to write perf metrics to file: {e}")


# ======== 导出器管理器：RequestMetricsExporterManager ========
class RequestMetricsExporterManager:
    """
    指标导出器管理器：支持同时启用多个导出目标

    Manager class for creating and managing RequestMetricsExporter instances.

    【核心功能】
    - 根据启动参数自动创建导出器（文件、数据库、Prometheus等）
    - 支持同时导出到多个目标（如同时写文件和推送到监控系统）
    - 统一调度所有导出器的写入操作
    - 支持私有插件扩展（通过 sglang.private 包）

    【使用示例】
    manager = RequestMetricsExporterManager(server_args)
    if manager.exporter_enabled():
        await manager.write_record(req_input, output_dict)
    """

    def __init__(
        self,
        server_args: ServerArgs,
        obj_skip_names: Optional[set[str]] = None,
        out_skip_names: Optional[set[str]] = None,
    ):
        self.server_args = server_args
        self.obj_skip_names = obj_skip_names or set()  # 输入对象中需要跳过的字段
        self.out_skip_names = out_skip_names or set()  # 输出字典中需要跳过的字段
        self._exporters: List[RequestMetricsExporter] = []  # 已启用的导出器列表
        self._create_exporters()  # 根据配置创建导出器

    def _create_exporters(self) -> None:
        """Create and configure RequestMetricsExporter instances based on server args."""
        # Create standard exporters
        self._exporters.extend(
            create_request_metrics_exporters(
                self.server_args, self.obj_skip_names, self.out_skip_names
            )
        )

        # Import additional RequestMetricsExporter from private fork if available; skip otherwise.
        try:
            from sglang.private.managers.request_metrics_exporter_factory import (
                create_private_request_metrics_exporters,
            )

            self._exporters.extend(
                create_private_request_metrics_exporters(
                    self.server_args, self.obj_skip_names, self.out_skip_names
                )
            )
        except ImportError:
            pass

    def exporter_enabled(self) -> bool:
        """Return true if at least one RequestMetricsExporter is enabled."""
        return len(self._exporters) > 0

    async def write_record(self, obj, out_dict: dict) -> None:
        """Write a record using all configured exporters."""
        for exporter in self._exporters:
            await exporter.write_record(obj, out_dict)


def create_request_metrics_exporters(
    server_args: ServerArgs,
    obj_skip_names: Optional[set[str]] = None,
    out_skip_names: Optional[set[str]] = None,
) -> List[RequestMetricsExporter]:
    """Create and configure `RequestMetricsExporter`s based on server args."""
    metrics_exporters = []

    if server_args.export_metrics_to_file:
        metrics_exporters.append(
            FileRequestMetricsExporter(server_args, obj_skip_names, out_skip_names)
        )

    return metrics_exporters
