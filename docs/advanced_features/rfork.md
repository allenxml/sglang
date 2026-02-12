# R-Fork

R-Fork (Tensor Remote Fork) is a novel weight loading methodology that leverages efficient inter-node GPU-to-GPU data transfer path to load tensors from a running SGLang instance to a new instance with zero-copy. It can significantly optimize the SGLang instance boot-up time by reducing model weights loading from several minutes to mere seconds.

To learn more details about R-Fork, please check **<a href=https://lmsys.org/blog/2025-12-10-rfork/> R-Fork blog </a>**

**中文对照**：# R-Fork

R-Fork（张量远程分叉）是一种新颖的权重加载方法论，它利用高效的节点间 GPU 到 GPU 数据传输路径，以零拷贝方式从正在运行的 SGLang 实例加载张量到新实例。它可以显著优化 SGLang 实例的启动时间，将模型权重加载从几分钟减少到几秒钟。

要了解关于 R-Fork 的更多详细信息，请查看 **<a href=https://lmsys.org/blog/2025-12-10-rfork/> R-Fork 博客 </a>**

## Usage

| Argument     | Usage                                      |
|--------------|--------------------------------------------|
| load-format  | set to `remote_instance` to enable R-Fork. |
| remote-instance-weight-loader-backend | `nccl` or `transfer_engine`, default value is `nccl` |
| remote-instance-weight-loader-seed-instance-ip | IP address of the seed instance who will provide the model weight |
| remote-instance-weight-loader-seed-instance-service-port | the port that the seed instance's HTTP server is listening on |
| remote-instance-weight-loader-send-weights-group-ports | the list of available ports on the seed instance that will be used to build NCCL communication groups between seed and client instance. This argument is only needed by `nccl` backend.  |
| remote-instance-weight-loader-start-seed-via-transfer-engine | set to start seed service that supports TransferEngine as backend. It is needed for seed instances when using `transfer_engine` as backend. |

### NCCL as backend

seed instance:
```shell
python -m sglang.launch_server [args]
```

client instance:
```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-send-weights-group-ports [send_weights_nccl_group_ports_list]  \
  --remote-instance-weight-loader-backend nccl
```

### TransferEngine as backend

seed instance:
```shell
python -m sglang.launch_server [args] \
  --remote-instance-weight-loader-start-seed-via-transfer-engine
```

```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-backend transfer_engine
```

## 代码实现
- **核心文件**: `python/sglang/srt/mem_cache/radix_cache.py`
- **架构**: 在 RadixCache 上下文中，分叉涉及分支前缀树，以允许多个请求或对话轮次共享公共历史记录而无需数据重复。这是通过节点分裂实现的，其中单个节点被分成一个父节点（共享前缀）和子节点（分歧分支）。
- **关键代码片段**:
  - `_split_node(self, key, child, split_len)`: 树分支的核心逻辑。它创建一个新的父节点，将键和 KV cache 值的共享部分移入其中，并将原始子节点链接到这个新父节点。
  - `_insert_helper(self, node, key, value, ...)`: 遍历树，并在找到部分匹配时触发 `_split_node`，有效地"分叉"路径以容纳新的 token 序列。
- **集成要点**: 此机制是 RadixAttention 系统的基础。它在 KV cache 插入（`insert`）和前缀匹配（`match_prefix`）期间自动触发，通过在不同的推理路径之间尽可能多地共享 KV cache 块来确保内存池的高效使用。
