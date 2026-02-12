# ================================================================================
# 📈 调度器指标混入类 (Scheduler Metrics Mixin)
# ================================================================================
#
# 【这个文件是什么】What This File Does
# 这个文件定义了 Scheduler 类的指标收集功能（使用 Mixin 设计模式），负责记录和导出
# 调度器运行时的各种性能指标，包括吞吐量、延迟、缓存命中率、GPU利用率等。
#
# 【生活比喻】Metaphor
# 想象这是一个"餐厅管理仪表盘"：
# - Scheduler = 餐厅管理员
# - SchedulerMetricsMixin = 仪表盘上的各种实时数据
# - metrics_collector = 数据收集器（记录每分钟服务了多少桌、平均等待时间等）
# - log_prefill_stats/log_decode_stats = 每次服务后更新仪表盘数据
#
# 【核心功能】Key Features
# 1. 吞吐量指标：input_throughput (Prefill TPS), gen_throughput (Decode TPS)
# 2. 资源利用率：GPU 显存占用、KV Cache 利用率、批次大小
# 3. 缓存效率：Radix Cache 命中率、新token比例
# 4. 延迟指标：队列等待时间、Forward Pass 时间
# 5. 分布式指标：DP/TP/PP 各rank的负载情况
#
# 【Mixin 设计模式】Design Pattern
# Mixin 是一种代码复用技术：
# - Scheduler 类继承 SchedulerMetricsMixin
# - 所有以 `self: Scheduler` 标注的方法都是 Scheduler 实例方法
# - 避免单个类过大（Scheduler 主类专注调度逻辑，指标收集分离到此文件）
#
# 【与 Prometheus 集成】Prometheus Integration
# - SchedulerMetricsCollector：将指标推送到 Prometheus
# - Grafana 可视化：通过 Prometheus 查询指标，绘制仪表盘
# - 告警规则：基于指标阈值触发告警（如 GPU 利用率 > 90%）
#
# ================================================================================

from __future__ import annotations

import dataclasses
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Optional, Union

from sglang.srt.disaggregation.kv_events import EventPublisherFactory, KVEventBatch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs  # 环境变量配置
from sglang.srt.managers.io_struct import (
    DisaggregationMetrics,
    GetLoadReqInput,
    GetLoadReqOutput,
    GetLoadsReqInput,
    GetLoadsReqOutput,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)
from sglang.srt.managers.scheduler import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.metrics.collector import (
    SchedulerMetricsCollector,
    SchedulerStats,
    compute_routing_key_stats,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.device_timer import DeviceTimer
from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import EmbeddingBatchResult, Scheduler

logger = logging.getLogger(__name__)

RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")
LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()
ENABLE_METRICS_DEVICE_TIMER = envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.get()


# ======== Prefill 阶段的统计数据结构 ========
@dataclasses.dataclass
class PrefillStats:
    """
    Prefill 批次的统计信息（用于日志和指标）

    Stats for logging prefill batch metrics.
    """
    log_input_tokens: int  # 本次 Prefill 处理的 token 总数
    log_hit_tokens: int  # 命中 RadixCache 的 token 数
    new_token_ratio: float  # 新token比例 = log_input_tokens / (log_input_tokens + log_hit_tokens)
    running_bs: int  # 当前运行的请求数（批次大小）
    num_new_seqs: int  # 本次新加入的序列数 len(can_run_list)


# ======== KV Cache 相关指标 ========
class KvMetrics:
    """KV Cache 相关的指标数据"""
    def __init__(self):
        self.request_active_slots = None  # 活跃的请求槽位数
        self.request_total_slots = None  # 总请求槽位数
        self.kv_active_blocks = None  # 活跃的 KV Cache 块数
        self.kv_total_blocks = None  # 总 KV Cache 块数
        self.num_requests_waiting = None  # 等待队列中的请求数
        self.gpu_cache_usage_perc = None  # GPU 缓存利用率（百分比）
        self.gpu_prefix_cache_hit_rate = None  # Prefix Cache 命中率
        self.data_parallel_rank = None  # 数据并行的 rank 编号


# ======== 调度器指标混入类 ========
class SchedulerMetricsMixin:
    def init_metrics(
        self: Scheduler, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ):
        """
        初始化调度器的指标收集系统

        【初始化内容】
        1. 基础统计变量（吞吐量、延迟等）
        2. Speculative Decoding 指标（如果启用）
        3. Prefill-Decode 分离指标（如果启用 Disaggregation）
        4. Prometheus 指标收集器（如果启用 metrics）
        """
        # ======== 基础统计变量 ========
        # Basic stats
        self.forward_ct_decode = 0  # Decode 阶段的 forward 次数
        self.num_generated_tokens = 0  # 累计生成的 token 数
        self.last_decode_stats_tic = time.perf_counter()  # 上次 Decode 统计的时间戳
        self.last_prefill_stats_tic = time.perf_counter()  # 上次 Prefill 统计的时间戳
        self.last_prefill_tokens = 0  # 上次 Prefill 处理的 token 数
        self.last_gen_throughput: float = 0.0  # 上次计算的生成吞吐量（token/s）
        self.last_input_throughput: float = 0.0  # 上次计算的输入吞吐量（token/s）
        self.step_time_dict = defaultdict(list)  # 记录每个批次大小的 step 时间：Dict[batch_size -> List[step_time]]

        # ======== Speculative Decoding 指标 ========
        # The number of accepted tokens and forward ct for the recent `decode_log_interval` batches (for logging)
        self.spec_num_accepted_tokens = 0  # 最近一段时间接受的推测 token 数
        self.spec_num_forward_ct = 0  # 最近一段时间的 forward 次数
        # The total number of accepted tokens and forward ct for the whole server lifetime
        self.spec_total_num_accepted_tokens = 0  # 服务器生命周期内总共接受的推测 token 数
        self.spec_total_num_forward_ct = 0  # 服务器生命周期内总共的 forward 次数

        # ======== Prefill-Decode 分离（PD Disaggregation）指标 ========
        # For PD disaggregation
        self.kv_transfer_speed_gb_s: float = 0.0  # KV Cache 传输速度（GB/s）
        self.kv_transfer_latency_ms: float = 0.0  # KV Cache 传输延迟（ms）
        self.kv_transfer_bootstrap_ms: float = 0.0  # 传输初始化时间（ms）
        self.kv_transfer_alloc_ms: float = 0.0  # 内存分配时间（ms）
        self.kv_transfer_total_mb: float = 0.0  # 传输总数据量（MB）

        # ======== 临时变量（用于跨方法传递信息）========
        # Only for `log_prefill_stats` to pass information to `log_prefill_stats_late`
        self.temp_prefill_info: Optional[Dict] = None

        # ======== 统计数据汇总对象 ========
        self.stats = SchedulerStats()

        # ======== Prometheus 指标收集器初始化 ========
        # Metrics
        self.current_scheduler_metrics_enabled = (
            self.attn_tp_rank == 0 or self.enable_metrics_for_all_schedulers
        )

        if self.enable_metrics:
            if self.server_args.disaggregation_mode == DisaggregationMode.PREFILL:
                engine_type = "prefill"
            elif self.server_args.disaggregation_mode == DisaggregationMode.DECODE:
                engine_type = "decode"
            else:
                engine_type = "unified"

            labels = {
                "model_name": self.server_args.served_model_name,
                "engine_type": engine_type,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "moe_ep_rank": self.moe_ep_rank,
            }
            if dp_rank is not None:
                labels["dp_rank"] = dp_rank
            if self.server_args.extra_metric_labels:
                labels.update(self.server_args.extra_metric_labels)
            self.metrics_collector = SchedulerMetricsCollector(
                labels=labels,
                enable_lora=self.enable_lora,
                server_args=self.server_args,
            )

            if ENABLE_METRICS_DEVICE_TIMER:
                self.forward_pass_device_timer = DeviceTimer(
                    reporter=self.metrics_collector.increment_gpu_execution_seconds,
                )

        if self.enable_kv_cache_events:
            self.init_kv_events(self.server_args.kv_events_config)

        self.scheduler_status_logger = SchedulerStatusLogger.maybe_create()

    def init_kv_events(self: Scheduler, kv_events_config: Optional[str]):
        if self.enable_kv_cache_events:
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config, self.attn_dp_rank
            )

    def update_spec_metrics(self: Scheduler, bs: int, num_accepted_tokens: int):
        self.spec_num_accepted_tokens += num_accepted_tokens + bs
        self.spec_num_forward_ct += bs
        self.num_generated_tokens += num_accepted_tokens

    def reset_metrics(self: Scheduler):
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.spec_num_accepted_tokens = 0
        self.spec_num_forward_ct = 0
        self.spec_total_num_accepted_tokens = 0
        self.spec_total_num_forward_ct = 0

    def log_prefill_stats(
        self: Scheduler,
        prefill_stats: PrefillStats,
        can_run_cuda_graph: bool,
    ):
        gap_latency = time.perf_counter() - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_input_throughput = self.last_prefill_tokens / gap_latency
        self.last_prefill_tokens = prefill_stats.log_input_tokens

        assert self.temp_prefill_info is None
        self.temp_prefill_info = dict(
            adder_log_input_tokens=prefill_stats.log_input_tokens,
            adder_log_hit_tokens=prefill_stats.log_hit_tokens,
        )

        # TODO: generalize this for various memory pools
        if self.is_hybrid_swa:
            (
                full_num_used,
                swa_num_used,
                full_token_usage,
                swa_token_usage,
                _,
                _,
                _,
                _,
            ) = self._get_swa_token_info()
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_token_usage, swa_token_usage)
            token_usage_msg = (
                f"full token usage: {full_token_usage:.2f}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        elif self.is_hybrid_ssm:
            (
                full_num_used,
                _,
                full_token_usage,
                mamba_usage,
                _,
                _,
                _,
                _,
            ) = self._get_mamba_token_info()
            num_used = full_num_used
            token_usage = full_token_usage
            token_usage_msg = (
                f"full token usage: {full_token_usage:.2f}, "
                f"mamba usage: {mamba_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_usage_msg = f"token usage: {token_usage:.2f}, "

        self.stats.new_token_ratio = prefill_stats.new_token_ratio
        iter_msg = f" [{self.forward_ct + 1}]" if LOG_FORWARD_ITERS else ""

        msg = (
            f"Prefill batch{iter_msg}, "
            f"#new-seq: {prefill_stats.num_new_seqs}, "
            f"#new-token: {prefill_stats.log_input_tokens}, "
            f"#cached-token: {prefill_stats.log_hit_tokens}, "
            f"{token_usage_msg}"
            f"#running-req: {prefill_stats.running_bs}, "
            f"#queue-req: {len(self.waiting_queue)}, "
        )

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            msg += f"#prealloc-req: {len(self.disagg_prefill_bootstrap_queue.queue)}, "
            msg += f"#inflight-req: {len(self.disagg_prefill_inflight_queue)}, "
            msg += f"input throughput (token/s): {self.last_input_throughput:.2f}, "
        else:
            msg += f"input throughput (token/s): {self.last_input_throughput:.2f}, "

        if self.server_args.language_only:
            msg += f"waiting-image-req: {len(self.mm_receiver.waiting_list)}, "
        graph_backend = defaultdict(
            lambda: "cuda graph",
            {
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )

        msg += f"{graph_backend[self.device]}: {can_run_cuda_graph}"

        logger.info(msg)

        if self.enable_metrics:
            # Basics
            total_tokens = prefill_stats.log_input_tokens + prefill_stats.log_hit_tokens
            cache_hit_rate = (
                prefill_stats.log_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )

            self.stats.num_running_reqs = prefill_stats.running_bs
            self.stats.num_running_reqs_offline_batch = 0
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = token_usage
            if self.is_hybrid_swa:
                self.stats.swa_token_usage = swa_token_usage
            if self.is_hybrid_ssm:
                self.stats.mamba_usage = mamba_usage
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            self.stats.cache_hit_rate = cache_hit_rate

            self.stats.max_total_num_tokens = self.max_total_num_tokens

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
                self.stats.kv_transfer_speed_gb_s = self.kv_transfer_speed_gb_s
                self.stats.kv_transfer_latency_ms = self.kv_transfer_latency_ms
                self.stats.kv_transfer_bootstrap_ms = self.kv_transfer_bootstrap_ms
                self.stats.kv_transfer_alloc_ms = self.kv_transfer_alloc_ms
                self.stats.kv_transfer_total_mb = self.kv_transfer_total_mb
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )

            # Others
            self.calculate_utilization()
            self.update_lora_metrics()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def log_prefill_stats_late(self: Scheduler, batch: Optional[ScheduleBatch]):
        """This should be called after `batch` has gathered enough metadata."""

        info = self.temp_prefill_info
        self.temp_prefill_info = None

        if self.enable_metrics and batch is not None and info is not None:
            self.metrics_collector.increment_realtime_tokens(
                prefill_compute_tokens=info["adder_log_input_tokens"],
                prefill_cache_tokens=info["adder_log_hit_tokens"],
                dp_cooperation_info=batch.dp_cooperation_info,
            )

    def log_decode_stats(
        self: Scheduler, can_run_cuda_graph: bool, running_batch: ScheduleBatch = None
    ):
        batch = running_batch or self.running_batch

        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency

        self.num_generated_tokens = 0
        num_running_reqs = len(batch.reqs)
        num_running_reqs_offline_batch = 0

        # TODO: generalize this for various memory pools
        if self.is_hybrid_swa:
            (
                full_num_used,
                swa_num_used,
                full_token_usage,
                swa_token_usage,
                _,
                _,
                _,
                _,
            ) = self._get_swa_token_info()
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_token_usage, swa_token_usage)
            token_usage_msg = (
                f"#full token: {full_num_used}, "
                f"full token usage: {full_token_usage:.2f}, "
                f"#swa token: {swa_num_used}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        elif self.is_hybrid_ssm:
            (
                full_num_used,
                mamba_used,
                full_token_usage,
                mamba_usage,
                _,
                _,
                _,
                _,
            ) = self._get_mamba_token_info()
            num_used = full_num_used
            token_usage = full_token_usage
            token_usage_msg = (
                f"#full token: {full_num_used}, "
                f"full token usage: {full_token_usage:.2f}, "
                f"mamba num: {mamba_used}, "
                f"mamba usage: {mamba_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_usage_msg = f"#token: {num_used}, token usage: {token_usage:.2f}, "

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.server_args.decode_log_interval
            )

        iter_msg = f" [{self.forward_ct}]" if LOG_FORWARD_ITERS else ""
        msg = f"Decode batch{iter_msg}, #running-req: {num_running_reqs}, {token_usage_msg}"

        if self.spec_algorithm.is_none():
            spec_accept_length = 0
            spec_accept_rate = 0
        else:
            spec_accept_length = (
                self.spec_num_accepted_tokens / self.spec_num_forward_ct
            )
            # Calculate acceptance rate: accepted tokens / total draft tokens
            draft_tokens_fallback = (self.server_args.speculative_num_steps or 0) + 1
            num_draft_tokens = (
                self.server_args.speculative_num_draft_tokens or draft_tokens_fallback
            )
            total_draft_tokens = self.spec_num_forward_ct * num_draft_tokens

            spec_accept_rate = (
                self.spec_num_accepted_tokens / total_draft_tokens
                if total_draft_tokens > 0
                else 0
            )
            self.spec_total_num_accepted_tokens += self.spec_num_accepted_tokens
            self.spec_total_num_forward_ct += self.spec_num_forward_ct
            self.spec_num_accepted_tokens = self.spec_num_forward_ct = 0
            msg += f"accept len: {spec_accept_length:.2f}, accept rate: {spec_accept_rate:.2f}, "
        cache_hit_rate = 0.0

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            msg += f"pre-allocated usage: {self.disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.max_total_num_tokens:.2f}, "
            msg += f"#prealloc-req: {len(self.disagg_decode_prealloc_queue.queue)}, "
            msg += f"#transfer-req: {len(self.disagg_decode_transfer_queue.queue)}, "
            msg += f"#retracted-req: {len(self.disagg_decode_prealloc_queue.retracted_queue)}, "

        if self.server_args.language_only:
            msg += f"waiting-image-req: {len(self.mm_receiver.waiting_list)}, "

        graph_backend = defaultdict(
            lambda: "cuda graph",
            {
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )
        msg += (
            f"{graph_backend[self.device]}: {can_run_cuda_graph}, "
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}"
        )

        logger.info(msg)
        if self.enable_metrics:
            # Basics
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_running_reqs_offline_batch = num_running_reqs_offline_batch
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = token_usage
            if self.is_hybrid_swa:
                self.stats.swa_token_usage = swa_token_usage
            if self.is_hybrid_ssm:
                self.stats.mamba_usage = mamba_usage
            self.stats.decode_sum_seq_lens = batch.seq_lens_cpu.sum().item()
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            self.stats.cache_hit_rate = cache_hit_rate

            self.stats.max_total_num_tokens = self.max_total_num_tokens

            # Speculative decoding
            self.stats.spec_accept_rate = spec_accept_rate
            self.stats.spec_accept_length = spec_accept_length

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )

            running_routing_keys = [r.routing_key for r in batch.reqs]
            waiting_routing_keys = [r.routing_key for r in self.waiting_queue]
            (
                self.stats.num_unique_running_routing_keys,
                self.stats.routing_key_running_req_counts,
            ) = compute_routing_key_stats(running_routing_keys)
            _, self.stats.routing_key_all_req_counts = compute_routing_key_stats(
                running_routing_keys + waiting_routing_keys
            )

            # Others
            self.calculate_utilization()
            self.update_lora_metrics()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def log_decode_stats_every_iteration(
        self: Scheduler, batch: ScheduleBatch, num_accepted_tokens: int
    ):
        if self.enable_metrics:
            self.metrics_collector.increment_realtime_tokens(
                # TODO unify this w/ the bumping logic in `Scheduler.num_generated_tokens` accumulator
                decode_tokens=batch.batch_size() + num_accepted_tokens,
                dp_cooperation_info=batch.dp_cooperation_info,
            )

        if x := self.scheduler_status_logger:
            x.maybe_dump(batch, self.waiting_queue)

    def log_batch_result_stats(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        if not self.enable_metrics:
            return
        if not isinstance(result, GenerationBatchResult):
            return

        if (m := result.expert_distribution_metrics) is not None:
            self.metrics_collector.increment_eplb_balancedness(
                forward_mode=batch.forward_mode.name.lower(),
                balancedness=m.eplb_balancedness.item(),
            )

    def _emit_kv_metrics(self: Scheduler):
        if not self.enable_kv_cache_events:
            return

        kv_metrics = KvMetrics()
        kv_metrics.request_active_slots = self.stats.num_running_reqs
        kv_metrics.request_total_slots = self.max_running_requests
        kv_metrics.kv_active_blocks = int(
            self.stats.token_usage * self.max_total_num_tokens
        )
        kv_metrics.kv_total_blocks = self.max_total_num_tokens
        kv_metrics.num_requests_waiting = self.stats.num_queue_reqs
        kv_metrics.gpu_cache_usage_perc = self.stats.token_usage
        kv_metrics.gpu_prefix_cache_hit_rate = self.stats.cache_hit_rate
        kv_metrics.data_parallel_rank = self.dp_rank if self.dp_rank is not None else 0

        if not self.send_metrics_from_scheduler.closed:
            self.send_metrics_from_scheduler.send_pyobj(kv_metrics)

    def _publish_kv_events(self: Scheduler):
        if not self.enable_kv_cache_events:
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

    def update_lora_metrics(self: Scheduler):
        """Update LoRA pool metrics for monitoring and autoscaling."""
        if not self.enable_lora:
            return

        try:
            # Get LoRA memory pool stats
            lora_manager = self.tp_worker.model_runner.lora_manager
            if lora_manager is None or lora_manager.memory_pool is None:
                return

            mem_pool = lora_manager.memory_pool
            slots_total = mem_pool.max_loras_per_batch

            # Calculate active adapters from running batch
            # This gives a true measure of current load for autoscaling purposes
            active_lora_ids = set()

            # For PP mode, check all running micro batches
            if hasattr(self, "running_mbs") and self.running_mbs:
                for batch in self.running_mbs:
                    if batch and hasattr(batch, "reqs"):
                        for req in batch.reqs:
                            if hasattr(req, "lora_id") and req.lora_id is not None:
                                active_lora_ids.add(req.lora_id)
            # For normal mode, check running_batch
            elif hasattr(self, "running_batch") and self.running_batch:
                if hasattr(self.running_batch, "reqs"):
                    for req in self.running_batch.reqs:
                        if hasattr(req, "lora_id") and req.lora_id is not None:
                            active_lora_ids.add(req.lora_id)

            # Count active adapters (excluding None for base model)
            slots_used = len(active_lora_ids)
            utilization = slots_used / slots_total if slots_total > 0 else 0.0

            # Update stats
            self.stats.lora_pool_slots_used = slots_used
            self.stats.lora_pool_slots_total = slots_total
            self.stats.lora_pool_utilization = utilization

        except Exception as e:
            logger.warning(f"Failed to update LoRA metrics: {e}")

    def calculate_utilization(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.utilization = -1
        else:
            if (
                self.stats.max_running_requests_under_SLO is not None
                and self.stats.max_running_requests_under_SLO > 0
            ):
                self.stats.utilization = max(
                    self.stats.num_running_reqs
                    / self.stats.max_running_requests_under_SLO,
                    self.stats.token_usage / 0.9,
                )

    def get_load(self: Scheduler, _: GetLoadReqInput = None) -> GetLoadReqOutput:
        if self.is_hybrid_swa:
            full_num_used, swa_num_used, *_ = self._get_swa_token_info()
            num_tokens = max(full_num_used, swa_num_used)
        elif self.is_hybrid_ssm:
            num_tokens = self._get_mamba_token_info()[0]
        else:
            num_tokens = self._get_token_info()[0]

        # Tokens in waiting queue, bootstrap queue, prealloc queue
        waiting_queues = [self.waiting_queue]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            waiting_queues.append(self.disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            waiting_queues.append(self.disagg_decode_prealloc_queue.queue)
            waiting_queues.append(self.disagg_decode_transfer_queue.queue)
            waiting_queues.append(self.disagg_decode_prealloc_queue.retracted_queue)

        num_tokens += sum(req.seqlen for queue in waiting_queues for req in queue)
        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)

        return GetLoadReqOutput(
            dp_rank=self.dp_rank,
            num_reqs=len(self.running_batch.reqs) + num_waiting_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_tokens=num_tokens,
            ts_tic=time.perf_counter(),
        )

    def get_loads(self: Scheduler, req: GetLoadsReqInput = None) -> GetLoadsReqOutput:
        """
        Get comprehensive load metrics for /v1/loads endpoint.

        Args:
            req: Request containing include list and optional dp_rank filter

        Returns:
            GetLoadsReqOutput with core metrics and optional detailed sections
        """
        if req is None:
            req = GetLoadsReqInput()

        include = set(req.include) if req.include else {"core"}
        include_all = "all" in include

        num_running_reqs = len(self.running_batch.reqs)

        waiting_queues = [self.waiting_queue]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            waiting_queues.append(self.disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            waiting_queues.append(self.disagg_decode_prealloc_queue.queue)
            waiting_queues.append(self.disagg_decode_transfer_queue.queue)
            waiting_queues.append(self.disagg_decode_prealloc_queue.retracted_queue)

        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)

        if self.is_hybrid_swa:
            full_num_used, swa_num_used, *_ = self._get_swa_token_info()
            num_used_tokens = max(full_num_used, swa_num_used)
        elif self.is_hybrid_ssm:
            num_used_tokens = self._get_mamba_token_info()[0]
        else:
            num_used_tokens = self._get_token_info()[0]

        token_usage = (
            num_used_tokens / self.max_total_num_tokens
            if self.max_total_num_tokens > 0
            else 0.0
        )

        memory = None
        if include_all or "memory" in include:
            try:
                memory = MemoryMetrics(
                    weight_gb=round(
                        self.tp_worker.model_runner.weight_load_mem_usage, 3
                    ),
                    kv_cache_gb=round(
                        self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                    ),
                    graph_gb=round(self.tp_worker.model_runner.graph_mem_usage, 3),
                    token_capacity=int(self.max_total_num_tokens),
                )
            except AttributeError as e:
                logger.debug(f"Memory metrics not available: {e}")

        speculative = None
        if include_all or "spec" in include:
            if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
                speculative = SpeculativeMetrics(
                    accept_length=(
                        self.spec_total_num_accepted_tokens
                        / self.spec_total_num_forward_ct
                    ),
                    accept_rate=self.stats.spec_accept_rate,
                )

        lora = None
        if include_all or "lora" in include:
            if hasattr(self, "lora_scheduler") and self.lora_scheduler is not None:
                lora = LoRAMetrics(
                    slots_used=self.stats.lora_pool_slots_used,
                    slots_total=self.stats.lora_pool_slots_total,
                    utilization=self.stats.lora_pool_utilization,
                )

        disaggregation = None
        if include_all or "disagg" in include:
            mode_str = "null"
            prefill_prealloc = 0
            prefill_inflight = 0
            decode_prealloc = 0
            decode_transfer = 0
            decode_retracted = 0

            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                mode_str = "prefill"
                prefill_prealloc = len(self.disagg_prefill_bootstrap_queue.queue)
                prefill_inflight = len(self.disagg_prefill_inflight_queue)
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                mode_str = "decode"
                decode_prealloc = len(self.disagg_decode_prealloc_queue.queue)
                decode_transfer = len(self.disagg_decode_transfer_queue.queue)
                decode_retracted = len(
                    self.disagg_decode_prealloc_queue.retracted_queue
                )

            disaggregation = DisaggregationMetrics(
                mode=mode_str,
                prefill_prealloc_queue_reqs=prefill_prealloc,
                prefill_inflight_queue_reqs=prefill_inflight,
                decode_prealloc_queue_reqs=decode_prealloc,
                decode_transfer_queue_reqs=decode_transfer,
                decode_retracted_queue_reqs=decode_retracted,
                kv_transfer_speed_gb_s=self.stats.kv_transfer_speed_gb_s,
                kv_transfer_latency_ms=self.stats.kv_transfer_latency_ms,
            )

        queues = None
        if include_all or "queues" in include:
            queues = QueueMetrics(
                waiting=len(self.waiting_queue),
                grammar=self.stats.num_grammar_queue_reqs,
                paused=self.stats.num_paused_reqs,
                retracted=self.stats.num_retracted_reqs,
            )

        return GetLoadsReqOutput(
            dp_rank=self.dp_rank,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_used_tokens=num_used_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            token_usage=round(token_usage, 4),
            gen_throughput=round(self.stats.gen_throughput, 2),
            cache_hit_rate=round(self.stats.cache_hit_rate, 4),
            utilization=round(self.stats.utilization, 4),
            max_running_requests=self.max_running_requests,
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
        )

    @contextmanager
    def record_forward_metrics(self: Scheduler, batch: ScheduleBatch):
        if not (self.enable_metrics and ENABLE_METRICS_DEVICE_TIMER):
            yield
            return

        category = "forward_" + batch.forward_mode.name.lower()
        with self.forward_pass_device_timer.wrap(
            metadata=dict(
                category=category,
                dp_cooperation_info=batch.dp_cooperation_info,
            ),
        ):
            yield
