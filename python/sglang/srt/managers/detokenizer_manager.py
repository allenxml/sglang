# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# ====================================================================
# 📚 学习导读：反分词管理器（DetokenizerManager）
# ====================================================================
#
# 【这个文件是做什么的？】
# DetokenizerManager 是推理流程的"最后一站"——它把 GPU 生成的
# Token ID 转换回人类可读的文字，并负责流式输出（streaming）。
#
# 【生活比喻】
# 就像餐厅的传菜员：
# - 厨师做好一道菜（GPU 生成 Token ID）
# - 传菜员把菜从后厨端到前厅（Token ID → 文字）
# - 如果是流式输出，就像一道一道上菜（每生成几个 Token 就发送一次）
# - 如果是非流式，就像等所有菜做完一起上（全部生成完再返回）
#
# 【核心职责】
# 1. 接收 Scheduler 发来的 Token ID 结果
# 2. 将 Token ID 转换为文字（反分词/Detokenization）
# 3. 处理流式输出（Stream）——边生成边返回
# 4. 检测生成是否完成（遇到 EOS 或达到 max_tokens）
# 5. 将完成的结果发送回 TokenizerManager
#
# 【流式输出原理】
# 非流式：等全部生成完 → 一次性返回完整文本
# 流式：  每生成几个Token → 立即转为文字 → 推送给用户
#         这就是 ChatGPT "打字效果" 的实现原理
#
# 【在推理流程中的位置】
# Scheduler(GPU计算完成) → 【DetokenizerManager】→ TokenizerManager → 用户
#
# 【阅读建议】
# 1. 先看 __init__() 了解初始化
# 2. 再看主循环方法了解如何处理每批结果
# 3. 关注流式输出的实现逻辑
# ====================================================================
"""DetokenizerManager is a process that detokenizes the token ids."""

import dataclasses
import logging
import os
import signal
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import psutil
import pybase64
import setproctitle
import zmq

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchMultimodalDecodeReq,
    BatchStrOutput,
    BatchTokenIDOutput,
    FreezeGCReq,
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiHttpWorkerDetokenizerMixin
from sglang.srt.metrics.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    freeze_gc,
    get_zmq_socket,
    kill_itself_when_parent_died,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import (
    TypeBasedDispatcher,
    find_printable_text,
    get_exception_traceback,
)

logger = logging.getLogger(__name__)

# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))


@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""

    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
    # Offset that's sent to tokenizer for incremental update.
    sent_offset: int = 0


# -----------------------------------------------------------------------
# 反分词管理器类：整个文件的核心
# 就像餐厅的"传菜部门"，负责把后厨（GPU）产出的半成品（Token ID）
# 加工成顾客能享用的成品（人类可读文字），并送到顾客桌上。
# 继承了 MultiHttpWorkerDetokenizerMixin，支持多个 HTTP Worker 并行工作。
# -----------------------------------------------------------------------
class DetokenizerManager(MultiHttpWorkerDetokenizerMixin):
    """DetokenizerManager is a process that detokenizes the token ids."""

    # -------------------------------------------------------------------
    # 初始化方法：搭建传菜部门的全部基础设施
    # 包括：通信通道（IPC）、分词器（tokenizer）、运行状态、请求分发器
    # -------------------------------------------------------------------
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Init inter-process communication
        self.init_ipc_channels(port_args)

        # Init tokenizer
        self.init_tokenizer(server_args)

        # Init running status
        self.init_running_status(server_args)

        if server_args.enable_metrics:
            start_cpu_monitor_thread("detokenizer")

        # Init dispatcher
        self.init_request_dispatcher()

    @staticmethod
    def is_health_check_request(rid: Optional[str]) -> bool:
        return isinstance(rid, str) and rid.startswith("HEALTH_CHECK")

    def init_ipc_channels(self, port_args: PortArgs):
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, False
        )

    def init_tokenizer(self, server_args: ServerArgs):
        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )

    def init_running_status(self, server_args: ServerArgs):
        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)
        self.is_dummy = False
        self.is_tool_call_parser_gpt_oss = server_args.tool_call_parser == "gpt-oss"
        self.disable_tokenizer_batch_decode = server_args.disable_tokenizer_batch_decode

        self.soft_watchdog = Watchdog.create(
            debug_name="DetokenizerManager",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DETOKENIZER.get(),
        )

    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOutput, self.handle_batch_embedding_out),
                (BatchTokenIDOutput, self.handle_batch_token_id_out),
                (BatchMultimodalDecodeReq, self.handle_multimodal_decode_req),
                (FreezeGCReq, self.handle_freeze_gc_req),
            ]
        )

    # -------------------------------------------------------------------
    # 主事件循环：传菜部门的"值班循环"
    # 不断从 Scheduler 接收新的 Token ID 批次，处理后发送给 TokenizerManager。
    # 就像传菜员一直守在出菜口，有菜就端，端完等下一道。
    # -------------------------------------------------------------------
    def event_loop(self):
        """The event loop that handles requests"""
        while True:
            with self.soft_watchdog.disable():
                recv_obj = self.recv_from_scheduler.recv_pyobj()
            output = self._request_dispatcher(recv_obj)
            if output is not None:
                self.send_to_tokenizer.send_pyobj(output)
            self.soft_watchdog.feed()

    # -------------------------------------------------------------------
    # 裁剪停止符：当生成遇到 stop 字符串或 stop token 时，
    # 把输出中多余的停止符部分去掉，保证返回给用户的文字是干净的。
    # 就像上菜前把盘子边缘擦干净。
    # -------------------------------------------------------------------
    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        if no_stop_trim or not finished_reason:
            return output

        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # TODO(lmzheng): handle the case where multiple stop strs are hit

        # Trim stop str.
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            return output[:pos] if pos != -1 else output

        # Trim stop token.
        if isinstance(matched, int) and isinstance(output, list):
            # 200012 <|call|> is the tool call token and one of eos tokens for gpt-oss model
            if output[-1] == 200012 and self.is_tool_call_parser_gpt_oss:
                return output
            assert len(output) > 0
            # NOTE: We can always assume the last token is the matched stop token
            return output[:-1]
        return output

    def handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOutput):
        # If it is embedding model, no detokenization is needed.
        return recv_obj

    # -------------------------------------------------------------------
    # 批量反分词：将一批 Token ID 列表转换为文字字符串
    # 按 (skip_special_tokens, spaces_between_special_tokens) 参数分组后
    # 调用 tokenizer.batch_decode() 批量解码，效率比逐条解码高得多。
    # 就像传菜员把去同一桌的菜归在一起端，减少来回跑的次数。
    # -------------------------------------------------------------------
    def _grouped_batch_decode(
        self,
        ids_list: List[List[int]],
        skip_list: List[bool],
        space_list: List[bool],
    ) -> List[str]:
        """Batch decode with grouping by (skip_special_tokens, spaces_between_special_tokens)."""

        assert self.tokenizer is not None

        # fast path
        first_skip, first_space = skip_list[0], space_list[0]
        if all(s == first_skip for s in skip_list) and all(
            sp == first_space for sp in space_list
        ):
            return self.tokenizer.batch_decode(
                ids_list,
                skip_special_tokens=first_skip,
                spaces_between_special_tokens=first_space,
            )

        # Group indices by (skip, space) tuple
        groups: Dict[Tuple[bool, bool], List[int]]
        groups = defaultdict(list)
        for idx, (skip, space) in enumerate(zip(skip_list, space_list)):
            groups[(skip, space)].append(idx)

        # Decode each group and collect results
        results: List[str] = [""] * len(ids_list)
        for (skip, space), indices in groups.items():
            decoded = self.tokenizer.batch_decode(
                [ids_list[idx] for idx in indices],
                skip_special_tokens=skip,
                spaces_between_special_tokens=space,
            )
            for idx, text in zip(indices, decoded):
                results[idx] = text

        return results

    # -------------------------------------------------------------------
    # 核心方法：处理一批 Token ID → 文字的完整转换流程
    # 这是整个文件最重要的方法，包含了：
    #   (1) 初始化或更新每个请求的解码状态（DecodeStatus）
    #   (2) 调用 batch_decode 将 Token ID 转换为文字
    #   (3) 增量解码（incremental decoding）——只发送新增的文字部分
    #       这是流式输出的关键：每次只计算"新增了什么文字"，避免重复发送
    #   (4) 处理生成结束（finished）——清理状态、裁剪 stop 字符串
    #
    # 流式输出的核心逻辑：
    #   - surr_offset / read_offset 记录上次已发送到哪里
    #   - 每次只解码新增的 Token，与之前的文字拼接
    #   - 用 find_printable_text() 确保不会发送半个汉字或半个 UTF-8 字符
    # -------------------------------------------------------------------
    def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):
        bs = len(recv_obj.rids)

        # Initialize decode status
        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]
            if rid not in self.decode_status:
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],
                    decode_ids=recv_obj.decode_ids[i],
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
                if not self.is_health_check_request(rid):
                    # for health check requests, we do not store the decode status
                    self.decode_status[rid] = s
            else:
                s = self.decode_status[rid]
                s.decode_ids.extend(recv_obj.decode_ids[i])

            read_ids.append(
                self.trim_matched_stop(
                    s.decode_ids[s.surr_offset :],
                    recv_obj.finished_reasons[i],
                    recv_obj.no_stop_trim[i],
                )
            )
            surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

        # Decode token ids to strings
        if not self.disable_tokenizer_batch_decode:
            if not self.is_dummy:
                # Run normal batch decode
                surr_texts = self._grouped_batch_decode(
                    surr_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
                read_texts = self._grouped_batch_decode(
                    read_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            else:
                # If it is dummy weights, just return dummy strings to prevent potential detokenization edge cases
                surr_texts = ["dog" for _ in surr_ids]
                read_texts = ["cat" for _ in read_ids]
        else:
            # Do not use batch decode to prevent some detokenization edge cases (e.g., gpt-oss).
            surr_texts = [
                self.tokenizer.decode(
                    surr, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for surr, skip, space in zip(
                    surr_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]
            read_texts = [
                self.tokenizer.decode(
                    read, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for read, skip, space in zip(
                    read_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]

        # Incremental decoding
        output_strs = []
        for i in range(bs):
            rid = recv_obj.rids[i]
            if self.is_health_check_request(rid):
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],
                    decode_ids=recv_obj.decode_ids[i],
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
            else:
                try:
                    s = self.decode_status[rid]
                except KeyError:
                    raise RuntimeError(
                        f"Decode status not found for request {rid}. "
                        "It may be due to the request being evicted from the decode status due to memory pressure. "
                        "Please increase the maximum number of requests by setting "
                        "the SGLANG_DETOKENIZER_MAX_STATES environment variable to a bigger value than the default value. "
                        f"The current value is {DETOKENIZER_MAX_STATES}. "
                        "For more details, see: https://github.com/sgl-project/sglang/issues/2812"
                    )
            new_text = read_texts[i][len(surr_texts[i]) :]
            if recv_obj.finished_reasons[i] is None:
                # Streaming chunk: update the decode status
                if len(new_text) > 0 and not new_text.endswith("�"):
                    s.decoded_text = s.decoded_text + new_text
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    new_text = ""
                else:
                    new_text = find_printable_text(new_text)
            else:
                if rid in self.decode_status:
                    del self.decode_status[rid]

            output_str = self.trim_matched_stop(
                s.decoded_text + new_text,
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )
            # Incrementally send text.
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            output_strs.append(incremental_output)

        return output_strs

    def _extract_routed_experts(
        self, recv_obj: BatchTokenIDOutput
    ) -> list[str | None] | None:
        routed_experts = None
        if recv_obj.routed_experts is not None:
            routed_experts = [
                (
                    pybase64.b64encode(routed_experts.numpy().tobytes()).decode("utf-8")
                    if routed_experts is not None
                    else None
                )
                for routed_experts in recv_obj.routed_experts
            ]
        return routed_experts

    # -------------------------------------------------------------------
    # 处理一批 Token ID 输出：将解码后的文字打包成 BatchStrOutput 返回
    # 这是从 Scheduler 收到 Token ID 后的入口方法，
    # 调用 _decode_batch_token_id_output() 完成反分词，
    # 然后把所有元信息（logprobs、token数量等）一起封装发回。
    # -------------------------------------------------------------------
    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
        # If handling idle batch, set output_strs to [].
        output_strs = (
            self._decode_batch_token_id_output(recv_obj)
            if len(recv_obj.rids) > 0
            else []
        )
        routed_experts = self._extract_routed_experts(recv_obj)

        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=recv_obj.output_ids,
            prompt_tokens=recv_obj.prompt_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            cached_tokens_details=recv_obj.cached_tokens_details,
            spec_verify_ct=recv_obj.spec_verify_ct,
            spec_accepted_tokens=recv_obj.spec_accepted_tokens,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=recv_obj.output_token_logprobs_val,
            output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            output_top_logprobs_val=recv_obj.output_top_logprobs_val,
            output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
            input_token_ids_logprobs_val=recv_obj.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=recv_obj.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=recv_obj.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=recv_obj.output_token_ids_logprobs_idx,
            output_token_entropy_val=recv_obj.output_token_entropy_val,
            output_hidden_states=recv_obj.output_hidden_states,
            routed_experts=routed_experts,
            customized_info=recv_obj.customized_info,
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=recv_obj.retraction_counts,
            token_steps=recv_obj.token_steps,
            load=recv_obj.load,
            queue_time=recv_obj.queue_time,
            forward_entry_time=recv_obj.forward_entry_time,
            prefill_launch_delay=recv_obj.prefill_launch_delay,
            prefill_launch_latency=recv_obj.prefill_launch_latency,
            prefill_finished_ts=recv_obj.prefill_finished_ts,
        )

    def handle_multimodal_decode_req(self, recv_obj: BatchMultimodalDecodeReq):
        raise NotImplementedError()

    def handle_freeze_gc_req(self, recv_req: FreezeGCReq):
        freeze_gc("Detokenizer Manager")
        return None


class LimitedCapacityDict(OrderedDict):
    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)
        # Set the new item
        super().__setitem__(key, value)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    detokenizer_manager_class=DetokenizerManager,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = detokenizer_manager_class(server_args, port_args)
        if server_args.tokenizer_worker_num == 1:
            manager.event_loop()
        else:
            manager.multi_http_worker_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        manager.maybe_clear_socket_mapping()
        parent_process.send_signal(signal.SIGQUIT)
