from __future__ import annotations

# ============================================================================
# 语法管理器 (Grammar Manager) - 结构化输出的守门人
# ============================================================================
#
# 【这是什么】Constrained/Structured Output Generation 的管理中心
# 强制模型按照特定语法规则生成输出（JSON schema、正则表达式、EBNF等）
#
# 【形象比喻】就像一位严格的编辑，确保作家（LLM模型）的作品严格遵循特定格式
# - 模型想输出什么 → 语法管理器检查是否符合规则 → 只有合规的内容才能通过
# - 比如要求输出JSON：模型不能随意发挥，必须生成有效的 {"key": "value"} 格式
#
# 【典型使用场景】
# 1. JSON生成: 强制模型输出符合schema的JSON对象（如API响应格式）
# 2. 正则表达式: 确保输出匹配特定模式（如邮箱格式、电话号码）
# 3. EBNF语法: 遵循自定义的语法规则（如特定领域语言DSL）
# 4. Structural tags: 生成带有特定结构标记的内容
#
# 【核心类】GrammarManager - 协调语法编译、缓存和跨进程同步
# ============================================================================

import logging
import time
from concurrent import futures
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    create_grammar_backend,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


# ============================================================================
# GrammarManager 类 - 语法协调中心
# ============================================================================
# 【核心职责】
# 1. 语法编译管理: 将JSON schema/regex/EBNF编译为可执行的状态机
# 2. 缓存协调: 避免重复编译相同的语法规则（编译很耗时）
# 3. 异步处理: 语法编译在后台进行，不阻塞主推理流程
# 4. 跨进程同步: 在分布式环境(DP/TP)中同步语法状态
#
# 【工作流程】
# 请求到达 → 检查缓存 → 未命中则异步编译 → 等待就绪 → 应用到生成过程
# ============================================================================
class GrammarManager:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.grammar_queue: List[Req] = []
        if not self.server_args.skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                self.server_args,
                scheduler.tokenizer,
                scheduler.model_config.vocab_size,
                scheduler.model_config.hf_eos_token_id,
            )
        else:
            self.grammar_backend = None

        self.grammar_sync_group = scheduler.dp_tp_cpu_group
        self.grammar_sync_size = scheduler.dp_tp_group.world_size
        self.grammar_sync_entry = scheduler.dp_tp_group.first_rank
        self.is_grammar_sync_entry = scheduler.dp_tp_group.is_first_rank

        self.SGLANG_GRAMMAR_POLL_INTERVAL = envs.SGLANG_GRAMMAR_POLL_INTERVAL.get()
        self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = (
            envs.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS.get()
        )

    def __len__(self):
        return len(self.grammar_queue)

    def clear(self):
        if self.grammar_backend:
            self.grammar_backend.reset()

    def has_waiting_grammars(self) -> bool:
        return len(self.grammar_queue) > 0

    def abort_requests(self, recv_req: AbortReq):
        for req in self.grammar_queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug(f"Abort grammar queue request. {req.rid=}")
                if req.grammar:
                    req.grammar.cancel()
                req.set_finish_with_abort("Aborted by AbortReq.")

    def process_req_with_grammar(self, req: Req) -> bool:
        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none"
                req.set_finish_with_abort(error_msg)
            else:
                if req.sampling_params.json_schema is not None:
                    key = ("json", req.sampling_params.json_schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key = ("ebnf", req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag:
                    key = ("structural_tag", req.sampling_params.structural_tag)

                value, cache_hit = self.grammar_backend.get_cached_or_future_value(
                    key, req.require_reasoning
                )
                req.grammar = value

                if not cache_hit:
                    req.grammar_key = key
                    add_to_grammar_queue = True
                else:
                    if value is INVALID_GRAMMAR_OBJ:  # We hit a cached invalid grammar.
                        error_msg = f"Invalid grammar request with cache hit: {key=}"
                        req.set_finish_with_abort(error_msg)

        if add_to_grammar_queue:
            self.grammar_queue.append(req)

        return add_to_grammar_queue

    def get_ready_grammar_requests(self) -> List[Req]:
        """
        Move requests whose grammar objects are ready from grammar_queue to waiting_queue.

        Rank i returns two sets ready_reqs_i, failed_reqs_i
        ready_reqs_all = all_gather(ready_reqs_i)
        failed_reqs_all = all_gather(failed_reqs_i)

        ready_reqs = intersect(ready_reqs_all)
        failed_reqs = union(failed_reqs_all)
        """
        ready_req_idxs: set[int] = set()
        failed_req_idxs: set[int] = set()

        # Poll for ready requests
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < self.SGLANG_GRAMMAR_POLL_INTERVAL:
            for i, req in enumerate(self.grammar_queue):
                if i in ready_req_idxs:
                    continue

                if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                    ready_req_idxs.add(i)
                    continue

                assert isinstance(req.grammar, futures.Future), f"{req=}"
                if req.grammar.done():
                    ready_req_idxs.add(i)

            # Sleep a bit to avoid busy waiting
            time.sleep(self.SGLANG_GRAMMAR_POLL_INTERVAL / 10)

        # Check failed requests
        for i, req in enumerate(self.grammar_queue):
            if i not in ready_req_idxs:
                self.grammar_queue[i].grammar_wait_ct += 1
                if (
                    self.grammar_queue[i].grammar_wait_ct
                    >= self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS
                ):
                    # Timeout after max poll iterations
                    # The actual waiting time is SGLANG_GRAMMAR_MAX_POLL_ITERATIONS * max(SGLANG_GRAMMAR_POLL_INTERVAL, GPU_forward_batch_latency)
                    failed_req_idxs.add(i)

        # Sync ready and failed requests across all ranks
        if self.grammar_sync_size == 1:
            synced_ready_req_idxs = ready_req_idxs
            synced_failed_req_idxs = failed_req_idxs
        else:
            all_gather_output = [None] * self.grammar_sync_size
            torch.distributed.all_gather_object(
                all_gather_output,
                (ready_req_idxs, failed_req_idxs),
                group=self.grammar_sync_group,
            )
            synced_ready_req_idxs = set.intersection(*[x[0] for x in all_gather_output])
            synced_failed_req_idxs = set.union(*[x[1] for x in all_gather_output])

        # Return ready requests
        return_reqs: List[Req] = []
        for i in synced_ready_req_idxs:
            req = self.grammar_queue[i]
            return_reqs.append(req)
            if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                continue

            req.grammar = req.grammar.result()
            self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
            if req.grammar is INVALID_GRAMMAR_OBJ:
                error_msg = f"Invalid grammar request: {req.grammar_key=}"
                req.set_finish_with_abort(error_msg)

        # Return failed requests
        for i in synced_failed_req_idxs:
            req = self.grammar_queue[i]
            return_reqs.append(req)

            req.grammar.cancel()
            self.grammar_backend.set_cache(req.grammar_key, INVALID_GRAMMAR_OBJ)
            error_msg = f"Grammar preprocessing timed out: {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)

        # Remove finished requests from grammar_queue
        self.grammar_queue = [
            req
            for i, req in enumerate(self.grammar_queue)
            if i not in synced_ready_req_idxs and i not in synced_failed_req_idxs
        ]
        return return_reqs
