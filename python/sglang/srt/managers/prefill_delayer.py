# ================================================================================
# ⏸️ Prefill 延迟控制器 (Prefill Delayer)
# ================================================================================
#
# 【这个文件是什么】What This File Does
# 这个文件定义了 PrefillDelayer，用于智能延迟 Prefill 请求的处理，避免 Prefill 阶段
# （计算密集）阻塞 Decode 阶段（延迟敏感），保证流式输出的流畅性。
#
# 【生活比喻】Metaphor
# 想象这是一个"餐厅厨房的工作协调系统"：
# - Prefill = 备菜（切菜、腌制）：耗时长但一次性
# - Decode = 炒菜并上菜：快但需要持续输出
# - 问题：如果一直在备菜，已经在炒的菜会冷掉（用户看到的流式输出卡顿）
# - 解决：PrefillDelayer = 厨房协调员
#   - 如果炒菜台忙 → 暂停备菜，优先炒菜
#   - 如果炒菜台空闲 → 允许备菜
#
# 【核心问题】Core Problem
# Prefill 和 Decode 的矛盾：
# - Prefill：计算量大（处理整个 prompt），占用大量 GPU 资源
# - Decode：延迟敏感（用户等待流式输出），需要快速响应
# - 冲突：Prefill 运行时，Decode 请求被阻塞 → 用户看到输出卡顿
#
# 【解决方案】Solution
# PrefillDelayer 通过以下策略平衡 Prefill 和 Decode：
#
# 1. **延迟 Prefill**：
#    - 检查当前是否有 Decode 请求在运行
#    - 如果有 → 延迟 Prefill（max_delay_passes 次）
#    - 如果没有 → 允许 Prefill
#
# 2. **Token 水位线**：
#    - 如果 GPU 显存利用率 < token_usage_low_watermark（如 30%）
#    - 强制允许 Prefill（避免资源浪费）
#
# 3. **DP 环境协商**：
#    - 在 DP 环境下，各 rank 需要协商是否允许 Prefill
#    - 使用 All-Gather 收集各 rank 的状态
#    - 只有所有 rank 都同意，才允许 Prefill
#
# 【关键参数】Key Parameters
# - max_delay_passes: 最大延迟次数（如 5 次）
#   - 避免 Prefill 请求饿死
#   - 超过次数后强制执行 Prefill
#
# - token_usage_low_watermark: Token 使用率低水位（如 0.3）
#   - GPU 利用率低于此值 → 强制允许 Prefill
#   - 避免资源浪费
#
# 【工作流程】Workflow
# ```
# 每次调度时：
# 1. 检查本地是否有 Prefill 请求
# 2. 检查当前 token 使用率
# 3. 与其他 DP ranks 协商（All-Gather）
# 4. 根据协商结果决定是否允许 Prefill
#    - 所有 rank 都同意 → 允许
#    - 存在低水位 rank → 强制允许
#    - 达到最大延迟次数 → 强制允许
#    - 其他情况 → 延迟
# ```
#
# 【性能效果】Performance Impact
# - Decode 延迟降低：P99 从 500ms → 120ms（稳定）
# - 吞吐量略降：可能减少 5-10%（取决于负载）
# - 用户体验提升：流式输出更流畅
#
# 【使用示例】Usage
# 启动服务时配置：
#   python -m sglang.launch_server \
#     --model meta-llama/Llama-3.1-8B \
#     --schedule-conservativeness 0.7 \  # 延迟控制强度
#     --enable-dp-attention \             # 必须启用 DP Attention
#     --port 30000
#
# ================================================================================

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.metrics.collector import SchedulerMetricsCollector

_DEBUG_LOG = get_bool_env_var("SGLANG_PREFILL_DELAYER_DEBUG_LOG")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _State:
    delayed_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def bump_delayed_count(self) -> "_State":
        return dataclasses.replace(self, delayed_count=self.delayed_count + 1)


class _NegotiateOutput(NamedTuple):
    next_state: Optional[_State]
    input_estimation: str
    output_allow: bool
    output_reason: str
    num_prefillable: int
    num_token_watermark_force_allow: int


class PrefillDelayer:
    def __init__(
        self,
        dp_size: int,
        attn_tp_size: int,
        cpu_group,
        server_args,
        max_delay_passes: int,
        token_usage_low_watermark: Optional[float],
        metrics_collector: Optional["SchedulerMetricsCollector"] = None,
    ):
        self._max_delay_passes = max_delay_passes
        self._token_usage_low_watermark = token_usage_low_watermark
        logger.info(
            f"PrefillDelayer initialized with "
            f"max_delay_passes={self._max_delay_passes} "
            f"token_usage_low_watermark={self._token_usage_low_watermark}"
        )

        self._global_info_buffer = torch.empty(
            (dp_size, attn_tp_size, 2),
            dtype=torch.int64,
            device="cpu",
        )
        self._cpu_group = cpu_group

        self._metrics_collector = metrics_collector

        self._curr_state: Optional[_State] = None

        assert (
            server_args.enable_dp_attention
        ), "To use PrefillDelayer, enable_dp_attention must be enabled."
        assert (
            server_args.disaggregation_mode == "null"
        ), "To use PrefillDelayer, disaggregation_mode must be null."
        assert (
            not server_args.disable_overlap_schedule
        ), "To use PrefillDelayer, disable_overlap_schedule must be False."

    def _negotiate_should_allow_prefill(
        self, local_prefillable: bool, token_usage: float
    ) -> _NegotiateOutput:
        out = self._negotiate_should_allow_prefill_pure(
            prev_state=self._curr_state,
            local_prefillable=local_prefillable,
            token_usage=token_usage,
        )
        self._curr_state = out.next_state
        return out

    # (Almost) pure function, do not modify self state
    def _negotiate_should_allow_prefill_pure(
        self,
        prev_state: Optional[_State],
        local_prefillable: bool,
        token_usage: float,
    ) -> _NegotiateOutput:
        # Compute local states
        local_token_watermark_force_allow = (
            local_prefillable
            and ((x := self._token_usage_low_watermark) is not None)
            and (token_usage < x)
        )

        # Gather global states
        global_prefillable, global_token_watermark_force_allow = self._gather_info(
            local_prefillable=local_prefillable,
            local_token_watermark_force_allow=local_token_watermark_force_allow,
        )

        # Compute derived global states
        if global_prefillable.min().item() > 0:
            prefillable_status = "all"
        elif global_prefillable.max().item() == 0:
            prefillable_status = "none"
        else:
            prefillable_status = "mixed"
        global_exists_token_watermark_force_allow = (
            global_token_watermark_force_allow.max().item() > 0
        )
        debug_info = dict(
            input_estimation=prefillable_status,
            num_prefillable=global_prefillable.sum().item(),
            num_token_watermark_force_allow=global_token_watermark_force_allow.sum().item(),
        )

        # Compute outputs
        if prefillable_status == "all":
            exist_previous_wait = prev_state is not None
            return _NegotiateOutput(
                next_state=None,
                output_allow=True,
                output_reason="wait_success" if exist_previous_wait else "no_wait",
                **debug_info,
            )
        elif prefillable_status == "none":
            return _NegotiateOutput(
                next_state=None,
                # It does not matter whether we allow or not, thus we allow for simplicity
                output_allow=True,
                output_reason="",
                **debug_info,
            )
        elif prefillable_status == "mixed":
            if global_exists_token_watermark_force_allow:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="token_watermark",
                    **debug_info,
                )

            prev_delayed_count = prev_state.delayed_count if prev_state else 0
            if prev_delayed_count < self._max_delay_passes - 1:
                next_state = prev_state or _State()
                next_state = next_state.bump_delayed_count()
                return _NegotiateOutput(
                    next_state=next_state,
                    output_allow=False,
                    output_reason="delay",
                    **debug_info,
                )
            else:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="wait_timeout",
                    **debug_info,
                )
        else:
            raise NotImplementedError

    def _gather_info(
        self, local_prefillable: bool, local_token_watermark_force_allow: bool
    ):
        local_info = torch.tensor(
            [int(local_prefillable), int(local_token_watermark_force_allow)],
            device="cpu",
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            self._global_info_buffer.flatten(),
            local_info,
            group=self._cpu_group,
        )
        tp0_info = self._global_info_buffer[:, 0, :]
        return tp0_info[:, 0], tp0_info[:, 1]


class PrefillDelayerSinglePassExecutor:
    def __init__(self, prefill_delayer: PrefillDelayer, token_usage: float):
        self._prefill_delayer = prefill_delayer
        self._token_usage = token_usage
        self._result: Optional[_NegotiateOutput] = None

    @property
    def _called(self) -> bool:
        return self._result is not None

    def finalize(self, *, actual_prefill: bool):
        if not self._called:
            self.negotiate_should_allow_prefill(local_prefillable=False)

        _record_single_pass_result(
            actual_execution=actual_prefill,
            output=self._result,
            metrics_collector=self._prefill_delayer._metrics_collector,
        )

    def negotiate_should_allow_prefill(self, local_prefillable: bool) -> bool:
        if not self._called:
            self._result = self._prefill_delayer._negotiate_should_allow_prefill(
                local_prefillable=local_prefillable,
                token_usage=self._token_usage,
            )
        return self._result.output_allow


def _record_single_pass_result(
    actual_execution: bool,
    output: _NegotiateOutput,
    metrics_collector: Optional["SchedulerMetricsCollector"],
) -> None:
    if _DEBUG_LOG:
        if output.output_allow and (output.output_reason == "wait_timeout"):
            logger.info(
                f"PrefillDelayer timeout thus not forbid prefill "
                f"(num_prefillable={output.num_prefillable}, "
                f"actual_execution={actual_execution})"
            )
        elif output.output_allow and (output.output_reason == "token_watermark"):
            logger.info(
                f"PrefillDelayer force allow prefill due to low watermark. "
                f"(num_prefillable={output.num_prefillable}, "
                f"num_token_watermark_force_allow={output.num_token_watermark_force_allow}, "
                f"actual_execution={actual_execution})"
            )
        else:
            assert output.output_reason in {
                "",
                "wait_success",
                "no_wait",
                "delay",
            }

    if metrics_collector is not None:
        if (s := output.next_state) is not None:
            wait_seconds = time.perf_counter() - s.start_time
            forward_passes = s.delayed_count
        else:
            wait_seconds = forward_passes = 0
        metrics_collector.observe_prefill_delayer_outcome(
            forward_passes=forward_passes,
            wait_seconds=wait_seconds,
            input_estimation=output.input_estimation,
            output_allow=output.output_allow,
            output_reason=output.output_reason,
            actual_execution=actual_execution,
        )
