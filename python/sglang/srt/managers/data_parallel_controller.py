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

# ================================================================================
# 🔀 数据并行控制器 (Data Parallel Controller)
# ================================================================================
#
# 【这个文件是什么】What This File Does
# 数据并行控制器（DataParallelController）负责在 DP（Data Parallelism）环境下，将用户请求
# 智能分发到多个独立的模型副本（DP Workers），实现横向扩展和负载均衡。
#
# 【生活比喻】Metaphor
# 想象这是一个"连锁餐厅的总调度中心"：
# - 总调度中心（DataParallelController） = 接听电话，决定把订单分配给哪家分店
# - 各分店（DP Workers） = 独立的模型副本，各自处理自己的订单
# - 负载均衡（Load Balancing） = 把订单分配给最空闲的分店，避免某家分店太忙
# - Round-Robin = 轮流分配（公平，但不考虑实际负载）
# - Least-Load = 分配给负载最低的分店（更智能）
#
# 【核心架构】Architecture
# 1. 请求路由：接收来自 TokenizerManager 的请求，选择目标 DP rank
# 2. 负载均衡：根据策略（Round-Robin / Least-Load）选择最优 Worker
# 3. 健康监控：定期检查各 Worker 的健康状态
# 4. 故障恢复：自动剔除故障 Worker，避免请求超时
#
# 【负载均衡策略】Load Balancing Methods
# - ROUND_ROBIN：轮流分配（简单，公平，但不考虑实际负载）
# - TOTAL_REQUESTS：分配给请求数最少的 Worker
# - TOTAL_TOKENS：分配给总 token 数最少的 Worker（推荐）
# - FOLLOW_BOOTSTRAP_ROOM：跟随 Prefill-Decode 分离模式的分配
#
# 【使用示例】Usage
# 启动 DP=3 的服务：
#   python -m sglang.launch_server \
#     --model meta-llama/Llama-3.1-70B \
#     --dp-size 3 \                      # 3个数据并行副本
#     --load-balance-method least_req \  # 负载均衡策略
#     --port 30000
#
# 【关键指标】Key Metrics
# - 各 Worker 的请求数和 token 数（用于负载均衡决策）
# - 请求分发延迟（Controller → Worker）
# - Worker 健康状态（存活/故障）
#
# ================================================================================

"""A controller that dispatches requests to multiple data parallel workers."""

import faulthandler
import logging
import multiprocessing as mp
import signal
import threading
import time
from enum import Enum, auto
from typing import Callable, List, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.environ import envs  # 环境变量配置
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    BlockReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.schedule_batch import Req, RequestStage
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.metrics.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.tracing.trace import (
    process_tracing_init,
    trace_get_proc_propagate_context,
    trace_set_proc_propagate_context,
    trace_set_thread_info,
    trace_slice_end,
    trace_slice_start,
)
from sglang.srt.utils import numa_utils
from sglang.srt.utils.common import (
    bind_port,
    configure_ipv6,
    configure_logger,
    get_zmq_socket,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


# ======== 负载均衡策略枚举 ========
class LoadBalanceMethod(Enum):
    """
    负载均衡方法

    Load balance method.

    【策略对比】
    - ROUND_ROBIN: 轮流分配（最简单，公平，但不考虑实际负载）
    - TOTAL_REQUESTS: 分配给当前请求数最少的 Worker
    - TOTAL_TOKENS: 分配给当前总 token 数最少的 Worker（推荐）
    - FOLLOW_BOOTSTRAP_ROOM: 跟随 Prefill-Decode 分离模式的分配
    """
    ROUND_ROBIN = auto()  # 轮询
    FOLLOW_BOOTSTRAP_ROOM = auto()  # 跟随预分配
    TOTAL_REQUESTS = auto()  # 最少请求数
    TOTAL_TOKENS = auto()  # 最少 token 数

    @classmethod
    def from_str(cls, method: str):
        """从字符串解析负载均衡策略"""
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


# ======== DP 预算管理器 ========
class DPBudget:
    """
    DP 预算管理器：跟踪各 DP Worker 的负载情况

    【核心数据】
    - total_requests[i]: 第 i 个 Worker 当前处理的请求数
    - total_tokens[i]: 第 i 个 Worker 当前处理的 token 总数

    【工作原理】
    - 定期从各 Worker 收集负载更新
    - 根据负载均衡策略选择目标 Worker
    - 分配请求后，乐观更新预算（避免连续分配到同一 Worker）
    """
    def __init__(self, dp_size: int):
        self.dp_size = dp_size  # DP 副本数量
        self.total_requests = [0] * dp_size  # 各 Worker 的请求数
        self.total_tokens = [0] * dp_size  # 各 Worker 的 token 总数

    def update_budget(self, load_update: WatchLoadUpdateReq):
        """
        更新各 Worker 的负载预算（从 Worker 收集的实际负载）

        Update the budget.
        """
        for load in load_update.loads:
            self.total_requests[load.dp_rank] = load.num_reqs
            self.total_tokens[load.dp_rank] = load.num_tokens

    def dispatch(self, method: LoadBalanceMethod):
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None

        # Increment the load of that worker by one as a heuristic
        self.total_requests[target_rank] += 1
        return target_rank


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        # For DP balance
        self.global_balance_id = 0

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Load balance budget
        self.dp_budget = DPBudget(server_args.dp_size)

        # To protect changing env vars to set CUDA_VISIBLE_DEVICES.
        self.env_lock = threading.Lock()

        # Launch data parallel workers
        self.scheduler_procs = []
        self.workers: List[zmq.Socket] = [None] * server_args.dp_size
        self.status: List[bool] = [True] * server_args.dp_size

        if server_args.enable_dp_attention:
            self.launch_dp_attention_schedulers(server_args, port_args)
            self.control_message_step = server_args.tp_size
        else:
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        self.init_dispatcher()

        self.soft_watchdog = Watchdog.create(
            debug_name="DataParallelController",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DP_CONTROLLER.get(),
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("data_parallel_controller")

    def send_to_all_workers(self, obj):
        for i, worker in enumerate(self.workers):
            if self.status[i]:
                worker.send_pyobj(obj)

    def send_control_message(self, obj):
        # Send control messages to first worker of tp group
        for worker in self.workers[:: self.control_message_step]:
            worker.send_pyobj(obj)

    def handle_load_update_req(self, obj):
        self.dp_budget.update_budget(obj)

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        self.status = ranks.status

    def dispatching_with_trace(self, req: Req):
        if self.server_args.enable_trace:
            trace_set_proc_propagate_context(req.rid, req.trace_context)
            trace_slice_start(RequestStage.DC_DISPATCH, req.rid)
            req.trace_context = trace_get_proc_propagate_context(req.rid)

        self.dispatching(req)

        if self.server_args.enable_trace:
            trace_slice_end(RequestStage.DC_DISPATCH, req.rid, thread_finish_flag=True)

    def init_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching_with_trace),
                (TokenizedEmbeddingReqInput, self.dispatching_with_trace),
                (BlockReqInput, self.send_to_all_workers),
                (WatchLoadUpdateReq, self.handle_load_update_req),
                (ActiveRanksOutput, self.update_active_ranks),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += (
                server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
            )

            if server_args.node_rank == 0:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        while True:
            time.sleep(30 * 24 * 3600)

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, worker_ports: Optional[List[int]] = None
    ) -> List[int]:
        """Broadcast worker ports from node 0 to all other nodes.

        Node 0 acts as the server, waiting for all other nodes to connect and
        sending them the pre-allocated worker ports. Other nodes act as clients,
        connecting to node 0 to receive their copy of the worker ports.

        Args:
            server_args: Server arguments containing node configuration.
            worker_ports: Pre-allocated worker ports to broadcast.

        Returns:
            List of worker ports (same on all nodes after broadcast).
        """
        # Determine the endpoint for inter-node communication
        if server_args.dist_init_addr is None:
            endpoint = f"tcp://127.0.0.1:{server_args.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA}"
        elif server_args.dist_init_addr.startswith("["):  # ipv6 address
            port, host = configure_ipv6(server_args.dist_init_addr)
            endpoint = f"tcp://{host}:{int(port) + DP_ATTENTION_HANDSHAKE_PORT_DELTA}"
        else:
            host, port = server_args.dist_init_addr.split(":")
            endpoint = f"tcp://{host}:{int(port) + DP_ATTENTION_HANDSHAKE_PORT_DELTA}"

        if server_args.node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            return self._broadcast_ports_as_server(
                endpoint, server_args.nnodes - 1, worker_ports
            )
        else:
            # Other nodes: Receive worker ports from node 0
            return self._receive_ports_as_client(endpoint, server_args.node_rank)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, worker_ports: List[int]
    ) -> List[int]:
        """Broadcast worker ports to all client nodes."""
        logger.debug(f"Broadcasting worker ports to {expected_clients} client nodes")
        logger.debug(f"Worker ports: {worker_ports}")

        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # Wait for client handshake
                client_rank = rep_socket.recv().decode()
                logger.debug(f"Received handshake from node {client_rank}")

                # Send worker ports to client
                rep_socket.send_pyobj(worker_ports)
                connected_clients += 1
                logger.debug(
                    f"Sent worker ports to {connected_clients}/{expected_clients} nodes"
                )

            logger.debug("Worker port broadcast completed")
            return worker_ports
        finally:
            rep_socket.close()

    def _receive_ports_as_client(self, endpoint: str, node_rank: int) -> List[int]:
        """Receive worker ports from the server node."""
        logger.debug(f"Connecting to node 0 to receive worker ports")

        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000)  # 10 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)

        try:
            # Send handshake with our node rank
            req_socket.send(str(node_rank).encode())

            # Receive worker ports
            worker_ports = req_socket.recv_pyobj()
            logger.debug(f"Received {len(worker_ports)} worker ports from node 0")
            return worker_ports
        except zmq.Again:
            logger.error("Timeout waiting for worker ports from node 0")
            raise RuntimeError(
                "Failed to receive worker ports from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        # Pre-allocate worker ports on node 0 to avoid conflicts
        worker_ports = []
        if server_args.node_rank == 0:
            for dp_rank in range(server_args.dp_size):
                port_and_socket = get_zmq_socket(self.context, zmq.PUSH)
                worker_ports.append(port_and_socket[0])
                self.workers[dp_rank] = port_and_socket[1]
                logger.debug(f"Assigned port {port_and_socket[0]} to worker {dp_rank}")

        broadcasted_ports = self._broadcast_worker_ports(
            server_args, worker_ports if worker_ports else None
        )
        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if server_args.enable_dp_attention:
                    # dp attention has different sharding logic
                    _, _, dp_rank = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                    )
                    # compute zmq ports for this dp rank
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    # Data parallelism reuses the tensor parallelism group,
                    # so all dp ranks should use the same nccl port.
                    rank_port_args.nccl_port = port_args.nccl_port

                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=self.run_scheduler_process_func,
                        args=(
                            server_args,
                            rank_port_args,
                            gpu_id,
                            tp_rank,
                            moe_ep_rank,
                            pp_rank,
                            dp_rank,
                            writer,
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess(), numa_utils.configure_subprocess(
                        server_args, gpu_id
                    ):
                        proc.start()
                self.scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def maybe_external_dp_rank_routing(self, req: Req):
        if req.data_parallel_rank is not None:
            logger.debug(f"Direct routing to DP rank {req.data_parallel_rank}")
            self.workers[req.data_parallel_rank].send_pyobj(req)
            return True
        return False

    def round_robin_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        while True:
            if self.status[self.round_robin_counter]:
                logger.debug(f"Choose worker {self.round_robin_counter}")
                self.workers[self.round_robin_counter].send_pyobj(req)
                self.round_robin_counter = (self.round_robin_counter + 1) % len(
                    self.workers
                )
                break
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

    def follow_bootstrap_room_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        # Set default bootstrap_room if in FAKE auto mode and room is None
        if (
            req.bootstrap_room is None
            and self.server_args.disaggregation_decode_enable_fake_auto
        ):
            req.bootstrap_room = self.round_robin_counter
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        target_rank = req.bootstrap_room % len(self.workers)
        self.workers[target_rank].send_pyobj(req)

    def total_requests_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.workers[target_worker].send_pyobj(req)

    def total_tokens_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS)
        self.workers[target_worker].send_pyobj(req)

    def event_loop(self):
        while True:
            while True:
                self.soft_watchdog.feed()
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    configure_logger(server_args)
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "DP Controller"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill DP Controller"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode DP Controller"
        trace_set_thread_info(thread_label)

    try:
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
