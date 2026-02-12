# Multi-Node Deployment

## Llama 3.1 405B

**Run 405B (fp16) on Two Nodes**

**中文对照**：在两个节点上运行 405B (fp16)

```bash
# replace 172.16.4.52:20000 with your own node ip address and port of the first node

python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tp 16 \
  --dist-init-addr 172.16.4.52:20000 \
  --nnodes 2 \
  --node-rank 0

python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tp 16 \
  --dist-init-addr 172.16.4.52:20000 \
  --nnodes 2 \
  --node-rank 1
```

Note that LLama 405B (fp8) can also be launched on a single node.

**中文对照**：请注意，Llama 405B (fp8) 也可以在单个节点上启动。

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8
```

## DeepSeek V3/R1

Please refer to [DeepSeek documents for reference](https://docs.sglang.io/basic_usage/deepseek.html#running-examples-on-multi-node).

**中文对照**：请参阅 [DeepSeek 文档](https://docs.sglang.io/basic_usage/deepseek.html#running-examples-on-multi-node) 作为参考。

## Multi-Node Inference on SLURM

This example showcases how to serve SGLang server across multiple nodes by SLURM. Submit the following job to the SLURM cluster.

**中文对照**：此示例展示了如何通过 SLURM 在多个节点上提供 SGLang 服务器服务。将以下作业提交到 SLURM 集群。

```
#!/bin/bash -l

#SBATCH -o SLURM_Logs/%x_%j_master.out
#SBATCH -e SLURM_Logs/%x_%j_master.err
#SBATCH -D ./
#SBATCH -J Llama-405B-Online-Inference-TP16-SGL

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1  # Ensure 1 task per node
#SBATCH --cpus-per-task=18
#SBATCH --mem=224GB
#SBATCH --partition="lmsys.org"
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

echo "[INFO] Activating environment on node $SLURM_PROCID"
if ! source ENV_FOLDER/bin/activate; then
    echo "[ERROR] Failed to activate environment" >&2
    exit 1
fi

# Define parameters
model=MODEL_PATH
tp_size=16

echo "[INFO] Running inference"
echo "[INFO] Model: $model"
echo "[INFO] TP Size: $tp_size"

# Set NCCL initialization address using the hostname of the head node
HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
NCCL_INIT_ADDR="${HEAD_NODE}:8000"
echo "[INFO] NCCL_INIT_ADDR: $NCCL_INIT_ADDR"

# Launch the model server on each node using SLURM
srun --ntasks=2 --nodes=2 --output="SLURM_Logs/%x_%j_node$SLURM_NODEID.out" \
    --error="SLURM_Logs/%x_%j_node$SLURM_NODEID.err" \
    python3 -m sglang.launch_server \
    --model-path "$model" \
    --grammar-backend "xgrammar" \
    --tp "$tp_size" \
    --dist-init-addr "$NCCL_INIT_ADDR" \
    --nnodes 2 \
    --node-rank "$SLURM_NODEID" &

# Wait for the NCCL server to be ready on port 30000
while ! nc -z "$HEAD_NODE" 30000; do
    sleep 1
    echo "[INFO] Waiting for $HEAD_NODE:30000 to accept connections"
done

echo "[INFO] $HEAD_NODE:30000 is ready to accept connections"

# Keep the script running until the SLURM job times out
wait
```

Then, you can test the server by sending requests following other [documents](https://docs.sglang.io/basic_usage/openai_api_completions.html).

**中文对照**：然后，您可以按照其他[文档](https://docs.sglang.io/basic_usage/openai_api_completions.html)发送请求来测试服务器。

Thanks for [aflah02](https://github.com/aflah02) for providing the example, based on his [blog post](https://aflah02.substack.com/p/multi-node-llm-inference-with-sglang).

**中文对照**：感谢 [aflah02](https://github.com/aflah02) 提供的示例，基于他的[博客文章](https://aflah02.substack.com/p/multi-node-llm-inference-with-sglang)。

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/server_args.py` | `--nnodes`、`--node-rank`、`--dist-init-addr` 命令行参数；校验 `tp_size` 能否被节点数整除 |
| `python/sglang/srt/distributed/parallel_state.py` | 进程组初始化：使用 `dist_init_addr` 在多节点间建立 NCCL 后端通信 |
| `python/sglang/srt/entrypoints/engine.py` | `_launch_subprocesses()`：在本地 GPU 上启动 TP worker 子进程；通过 NCCL 与远程节点协调 |
| `python/sglang/srt/managers/tp_worker.py` | `TpWorker`：每个 GPU worker 加入分布式组；全局 rank = `node_rank * gpus_per_node + local_rank` |

### 集成要点

- **NCCL 初始化**：`--dist-init-addr <ip>:<port>` 设置汇合地址；所有节点必须指向同一地址（通常为节点 0）
- **跨节点 TP**：`--tp 16 --nnodes 2` 表示每节点 8 块 GPU；模型层通过张量并行分片到全部 16 块 GPU 上
- **SLURM 集成**：`$SLURM_NODEID` 对应 `--node-rank`；头节点主机名用作 `--dist-init-addr`
- **FP8 单节点**：Llama 405B FP8 可在单个 8-GPU 节点运行，避免多节点通信开销
