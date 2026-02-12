# DP for Multi-Modal Encoder in SGLang

A typical VLM architecture involves two main components: an multi-modal encoder and a text decoder.

Most VLMs utilize a Vision Transformer (ViT) as their multi-modal encoder, it is responsible for processing visual data, extracting features (objects, colors, textures, etc.), and transforming them into a format that can be understood by the model.

The text decoder is based on LLM. It processes textual data and generates output based on the encoded visual features.

However, since the size of ViT is very small compared to language decoders,
there is relatively little gain from TP. On the other hand, TP incurs significant communication
overhead because of all-reduce being performed after every layer.

Placing the ViT in data parallel while keeping the LLM in tensor parallel consistently lowers TTFT and boosts end-to-end throughput. In this hybrid layout, the vision front-end becomes parallel and lightweight, while scarce interconnect bandwidth and collective ops are reserved for the LLM.

Data parallelism replicates the entire model across multiple GPU sets and processes different batches of requests in parallel.

## Command Example
You can enable batch-level DP by setting `mm-enable-dp-encoder`, for example:
```
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --tp 2 \
    --mm-enable-dp-encoder
```

## Known supported models
- Qwen2.5-VL (<https://github.com/sgl-project/sglang/pull/13126>)
- Qwen3-VL (<https://github.com/sgl-project/sglang/pull/13724>)
- InternVL (<https://github.com/sgl-project/sglang/pull/13925>)
- GLM-4.5V & GLM-4.6V (<https://github.com/sgl-project/sglang/pull/14097>)

## 代码实现
- **核心文件**: `python/sglang/srt/multimodal/mm_utils.py`
- **架构**: 多模态编码器的数据并行实现为混合布局，其中视觉Transformer（ViT）使用数据并行（DP），而LLM解码器保持张量并行（TP）。这种设计显著减少了在纯TP配置中每个ViT层后执行all-reduce操作所产生的通信开销。
- **关键代码片段**:
  - `get_dp_encoder_lb_assignment(sizes, num_gpus)`: 实现了一个贪婪负载均衡算法，基于总补丁数而非仅图像数量在GPU之间分配图像。
  - `run_dp_sharded_vision_model(...)` 和 `run_dp_sharded_mrope_vision_model(...)`: 协调视觉模型的分片执行。它们处理输入张量的分片、执行本地前向传递，并执行 `tensor_model_parallel_all_gather` 以在各rank之间同步embeddings。
- **集成要点**: 通过 `--mm-enable-dp-encoder` 标志激活。系统根据TP大小自动计算DP大小。它通过拦截视觉输入并在到达语言模型解码器之前将其路由到均衡的DP处理函数，集成到模型的前向路径中。
