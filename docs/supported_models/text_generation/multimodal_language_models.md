# Multimodal Language Models

These models accept multi-modal inputs (e.g., images and text) and generate text output. They augment language models with multimodal encoders.

**中文对照**：多模态语言模型

这些模型接受多模态输入（例如，图像和文本）并生成文本输出。它们通过多模态编码器增强语言模型。

## Example launch Command

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \  # example HF/local path
  --host 0.0.0.0 \
  --port 30000 \
```

> See the [OpenAI APIs section](https://docs.sglang.io/basic_usage/openai_api_vision.html) for how to send multimodal requests.

**中文对照**：示例启动命令

请参阅 [OpenAI API 部分](https://docs.sglang.io/basic_usage/openai_api_vision.html) 了解如何发送多模态请求。

## Supported models

Below the supported models are summarized in a table.

If you are unsure if a specific architecture is implemented, you can search for it via GitHub. For example, to search for `Qwen2_5_VLForConditionalGeneration`, use the expression:

```
repo:sgl-project/sglang path:/^python\/sglang\/srt\/models\// Qwen2_5_VLForConditionalGeneration
```

in the GitHub search bar.

**中文对照**：支持的模型

下表总结了支持的模型。

如果您不确定某个特定架构是否已实现，您可以通过 GitHub 搜索。


| Model Family (Variants)    | Example HuggingFace Identifier             | Description                                                                                                                                                                                                     | Notes |
|----------------------------|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| **Qwen-VL** | `Qwen/Qwen3-VL-235B-A22B-Instruct`              | Alibaba's vision-language extension of Qwen; for example, Qwen2.5-VL (7B and larger variants) can analyze and converse about image content.                                                                     |  |
| **DeepSeek-VL2**           | `deepseek-ai/deepseek-vl2`                 | Vision-language variant of DeepSeek (with a dedicated image processor), enabling advanced multimodal reasoning on image and text inputs.                                                                        |  |
| **DeepSeek-OCR / OCR-2**   | `deepseek-ai/DeepSeek-OCR-2`               | OCR-focused DeepSeek models for document understanding and text extraction.                                                                                                                                    | Use `--trust-remote-code`. |
| **Janus-Pro** (1B, 7B)     | `deepseek-ai/Janus-Pro-7B`                 | DeepSeek's open-source multimodal model capable of both image understanding and generation. Janus-Pro employs a decoupled architecture for separate visual encoding paths, enhancing performance in both tasks. |  |
| **MiniCPM-V / MiniCPM-o**  | `openbmb/MiniCPM-V-2_6`                    | MiniCPM-V (2.6, ~8B) supports image inputs, and MiniCPM-o adds audio/video; these multimodal LLMs are optimized for end-side deployment on mobile/edge devices.                                                 |  |
| **Llama 3.2 Vision** (11B) | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Vision-enabled variant of Llama 3 (11B) that accepts image inputs for visual question answering and other multimodal tasks.                                                                                     |  |
| **LLaVA** (v1.5 & v1.6)    | *e.g.* `liuhaotian/llava-v1.5-13b`         | Open vision-chat models that add an image encoder to LLaMA/Vicuna (e.g. LLaMA2 13B) for following multimodal instruction prompts.                                                                               |  |
| **LLaVA-NeXT** (8B, 72B)   | `lmms-lab/llava-next-72b`                  | Improved LLaVA models (with an 8B Llama3 version and a 72B version) offering enhanced visual instruction-following and accuracy on multimodal benchmarks.                                                       |  |
| **LLaVA-OneVision**        | `lmms-lab/llava-onevision-qwen2-7b-ov`     | Enhanced LLaVA variant integrating Qwen as the backbone; supports multiple images (and even video frames) as inputs via an OpenAI Vision API-compatible format.                                                 |  |
| **Gemma 3 (Multimodal)**   | `google/gemma-3-4b-it`                     | Gemma 3's larger models (4B, 12B, 27B) accept images (each image encoded as 256 tokens) alongside text in a combined 128K-token context.                                                                        |  |
| **Kimi-VL** (A3B)          | `moonshotai/Kimi-VL-A3B-Instruct`          | Kimi-VL is a multimodal model that can understand and generate text from images.                                                                                                                                |  |
| **Mistral-Small-3.1-24B**  | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral 3.1 is a multimodal model that can generate text from text or images input. It also supports tool calling and structured output. |  |
| **Phi-4-multimodal-instruct**  | `microsoft/Phi-4-multimodal-instruct` | Phi-4-multimodal-instruct is the multimodal variant of the Phi-4-mini model, enhanced with LoRA for improved multimodal capabilities. It supports text, vision and audio modalities in SGLang. |  |
| **MiMo-VL** (7B)           | `XiaomiMiMo/MiMo-VL-7B-RL`                 | Xiaomi's compact yet powerful vision-language model featuring a native resolution ViT encoder for fine-grained visual details, an MLP projector for cross-modal alignment, and the MiMo-7B language model optimized for complex reasoning tasks. |  |
| **GLM-4.5V** (106B) /  **GLM-4.1V**(9B)           | `zai-org/GLM-4.5V`                   | GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning                                                                                                                                                                                                      | Use `--chat-template glm-4v` |
| **GLM-OCR**          | `zai-org/GLM-OCR`                   | GLM-OCR: A fast and accurate general OCR model                                                                   |  |
| **DotsVLM** (General/OCR)  | `rednote-hilab/dots.vlm1.inst`             | RedNote's vision-language model built on a 1.2B vision encoder and DeepSeek V3 LLM, featuring NaViT vision encoder trained from scratch with dynamic resolution support and enhanced OCR capabilities through structured image data training. |  |
| **DotsVLM-OCR**            | `rednote-hilab/dots.ocr`                   | Specialized OCR variant of DotsVLM optimized for optical character recognition tasks with enhanced text extraction and document understanding capabilities. | Don't use `--trust-remote-code` |
| **NVILA** (8B, 15B, Lite-2B, Lite-8B, Lite-15B) | `Efficient-Large-Model/NVILA-8B` | `chatml` | NVILA explores the full stack efficiency of multi-modal design, achieving cheaper training, faster deployment and better performance. |
| **NVIDIA Nemotron Nano 2.0 VL** | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | NVIDIA Nemotron Nano v2 VL enables multi-image reasoning and video understanding, along with strong document intelligence, visual Q&A and summarization capabilities. It builds on Nemotron Nano V2, a hybrid Mamba-Transformer LLM, in order to achieve higher inference throughput in long document and video scenarios. | Use `--trust-remote-code`. You may need to adjust `--max-mamba-cache-size` [default is 512] to fit memory constraints. |
| **Ernie4.5-VL** | `baidu/ERNIE-4.5-VL-28B-A3B-PT`              | Baidu's vision-language models(28B,424B). Support image and video comprehension, and also support thinking.                                                                     |  |
| **JetVLM** |  | JetVLM is an vision-language model designed for high-performance multimodal understanding and generation tasks built upon Jet-Nemotron. | Coming soon |
| **Step3-VL** (10B) | `stepfun-ai/Step3-VL-10B` | StepFun's lightweight open-source 10B parameter VLM for multimodal intelligence, excelling in visual perception, complex reasoning, and human alignment. |  |
| **Qwen3-Omni** | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |  Alibaba's omni-modal MoE model. Currently supports the **Thinker** component (multimodal understanding for text, images, audio, and video), while the **Talker** component (audio generation) is not yet supported. |  |

## Video Input Support

SGLang supports video input for Vision-Language Models (VLMs), enabling temporal reasoning tasks such as video question answering, captioning, and holistic scene understanding. Video clips are decoded, key frames are sampled, and the resulting tensors are batched together with the text prompt, allowing multimodal inference to integrate visual and linguistic context.

**中文对照**：视频输入支持

| Model Family | Example Identifier | Video notes |
|--------------|--------------------|-------------|
| **Qwen-VL** (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-Omni) | `Qwen/Qwen3-VL-235B-A22B-Instruct` | The processor gathers `video_data`, runs Qwen's frame sampler, and merges the resulting features with text tokens before inference. |
| **GLM-4v** (4.5V, 4.1V, MOE) | `zai-org/GLM-4.5V` | Video clips are read with Decord, converted to tensors, and passed to the model alongside metadata for rotary-position handling. |
| **NVILA** (Full & Lite) | `Efficient-Large-Model/NVILA-8B` | The runtime samples eight frames per clip and attaches them to the multimodal request when `video_data` is present. |
| **LLaVA video variants** (LLaVA-NeXT-Video, LLaVA-OneVision) | `lmms-lab/LLaVA-NeXT-Video-7B` | The processor routes video prompts to the LlavaVid video-enabled architecture, and the provided example shows how to query it with `sgl.video(...)` clips. |
| **NVIDIA Nemotron Nano 2.0 VL** | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | The processor samples at 2 FPS, at a max of 128 frames, as per model training. The model uses [EVS](../../python/sglang/srt/multimodal/evs/README.md), a pruning method that removes redundant tokens from video embeddings. By default `video_pruning_rate=0.7`. Change this by providing: `--json-model-override-args '{"video_pruning_rate": 0.0}'` to disable EVS, for example. |
| **JetVLM** |  | The runtime samples eight frames per clip and attaches them to the multimodal request when `video_data` is present. |

Use `sgl.video(path, num_frames)` when building prompts to attach clips from your SGLang programs.

**中文对照**：使用 `sgl.video(path, num_frames)` 在构建提示时附加片段。

Example OpenAI-compatible request that sends a video clip:

**中文对照**：发送视频剪辑的 OpenAI 兼容请求示例：

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s happening in this video?"},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/jobs_presenting_ipod.mp4"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print(response.text)
```

## Usage Notes

### Performance Optimization

For multimodal models, you can use the `--keep-mm-feature-on-device` flag to optimize for latency at the cost of increased GPU memory usage:

- **Default behavior**: Multimodal feature tensors are moved to CPU after processing to save GPU memory
- **With `--keep-mm-feature-on-device`**: Feature tensors remain on GPU, reducing device-to-host copy overhead and improving latency, but consuming more GPU memory

Use this flag when you have sufficient GPU memory and want to minimize latency for multimodal inference.

**中文对照**：使用说明

### 性能优化

### Multimodal Inputs Limitation

- **Use `--mm-process-config '{"image":{"max_pixels":1048576},"video":{"fps":3,"max_pixels":602112,"max_frames":60}}'`**: To set `image`, `video`, and `audio` input limits.

This can reduce GPU memory usage, improve inference speed, and help to avoid OOM, but may impact model performance, thus set a proper value based on your specific use case. Currently, only `qwen_vl` supports this config. Please refer to [qwen_vl processor](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/multimodal/processors/qwen_vl.py) for understanding the meaning of each parameter.

**中文对照**：多模态输入限制

### Bidirectional Attention in Multimodal Model Serving
**Note for serving the Gemma-3 multimodal model**:

As mentioned in [Welcome Gemma 3: Google's all new multimodal, multilingual, long context open LLM
](https://huggingface.co/blog/gemma3#multimodality), Gemma-3 employs bidirectional attention between image tokens during the prefill phase. Currently, SGLang only supports bidirectional attention when using the Triton Attention Backend. Note, however, that SGLang's current bidirectional attention implementation is incompatible with both CUDA Graph and Chunked Prefill.

To enable bidirectional attention, you can use the `TritonAttnBackend` while disabling CUDA Graph and Chunked Prefill. Example launch command:
```shell
python -m sglang.launch_server \
  --model-path google/gemma-3-4b-it \
  --host 0.0.0.0 --port 30000 \
  --enable-multimodal \
  --dtype bfloat16 --triton-attention-reduce-in-fp32 \
  --attention-backend triton \ # Use Triton attention backend
  --disable-cuda-graph \ # Disable Cuda Graph
  --chunked-prefill-size -1 # Disable Chunked Prefill
```

If higher serving performance is required and a certain degree of accuracy loss is acceptable, you may choose to use other attention backends, and you can also enable features like CUDA Graph and Chunked Prefill for better performance, but note that the model will fall back to using causal attention instead of bidirectional attention.

**中文对照**：多模态模型服务中的双向注意力

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/models/` | 多模态模型实现：如 `qwen2_vl.py`、`llava.py`、`gemma3.py` 等 |
| `python/sglang/srt/multimodal/processors/` | 多模态处理器：每个 VLM 系列对应一个 `BaseMultimodalProcessor` 子类，处理图像/视频预处理 |
| `python/sglang/srt/multimodal/mm_utils.py` | 多模态工具函数：图像解码、视频帧采样、特征张量管理 |
| `python/sglang/srt/multimodal/vit_cuda_graph_runner.py` | ViT CUDA Graph 运行器：为视觉编码器捕获 CUDA Graph 以加速推理 |
| `python/sglang/srt/configs/model_config.py` | `is_multimodal_model()`：判断模型是否为多模态类型 |

### 集成要点

- **启用多模态**：部分模型需要 `--enable-multimodal` 标志或 `--chat-template` 参数
- **图像特征缓存**：`--keep-mm-feature-on-device` 保持特征在 GPU 上以降低延迟（增加显存占用）
- **输入限制**：通过 `--mm-process-config` 设置图像/视频的最大分辨率和帧数限制
- **视频支持**：支持 Qwen-VL、GLM-4v、NVILA、LLaVA 等系列的视频输入，通过 `video_url` 字段传入
- **双向注意力**：Gemma-3 多模态需使用 Triton 注意力后端，并禁用 CUDA Graph 和 Chunked Prefill
