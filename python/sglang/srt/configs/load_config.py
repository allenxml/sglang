# ================================================================================
# 📦 模型加载配置 (Load Config)
# ================================================================================
#
# 【这个文件是什么】What This File Does
# 这个文件定义了模型权重加载的配置类（LoadConfig），控制如何从磁盘/HuggingFace Hub
# 加载模型权重，支持多种格式（safetensors、pytorch、GGUF、量化格式等）。
#
# 【生活比喻】Metaphor
# 想象这是一个"图书馆书籍提取规则"：
# - LoadConfig = 图书管理员的工作手册
# - load_format = 书籍格式（精装本、电子版、有声书等）
# - download_dir = 书库位置
# - decryption_key = 加密书籍的密钥
#
# 【核心配置】Key Configurations
# 1. load_format: 权重文件格式
#    - auto: 自动检测（优先 safetensors，回退到 pt）
#    - safetensors: HuggingFace 推荐格式（安全、快速）
#    - pt: PyTorch 原生格式（.bin 文件）
#    - gguf: llama.cpp 格式（量化模型）
#    - bitsandbytes: NF4/INT8 量化格式
#
# 2. download_dir: 模型权重下载/缓存目录
#    - 默认：~/.cache/huggingface/hub
#    - 可自定义（如挂载的 NFS 共享目录）
#
# 3. model_loader_extra_config: 额外加载参数（JSON 格式）
#    - 用于特殊模型的自定义加载逻辑
#
# 4. 量化配置:
#    - modelopt_config: ModelOpt 量化配置
#    - rl_quant_profile: RL 量化 profile 文件路径
#
# 【使用示例】Usage
# 加载 AWQ 量化模型：
#   python -m sglang.launch_server \
#     --model meta-llama/Llama-3.1-70B-Instruct-AWQ \
#     --load-format auto \
#     --download-dir /mnt/models
#
# ================================================================================

# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/config.py
import enum
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import orjson

from sglang.srt.configs.modelopt_config import ModelOptConfig
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)


# ======== 模型权重格式枚举 ========
class LoadFormat(str, enum.Enum):
    """
    模型权重加载格式

    【常用格式】
    - AUTO: 自动检测（推荐）
    - SAFETENSORS: HuggingFace 推荐格式（安全、高效）
    - PT: PyTorch 原生格式（.bin 文件）
    - GGUF: llama.cpp 量化格式
    - BITSANDBYTES: NF4/INT8 量化

    【特殊格式】
    - DUMMY: 随机初始化权重（用于性能测试）
    - NPCACHE: PyTorch + NumPy 缓存（加速重复加载）
    - REMOTE: 远程权重加载（跨节点）
    """
    AUTO = "auto"  # 自动检测
    PT = "pt"  # PyTorch 格式（.bin）
    SAFETENSORS = "safetensors"  # SafeTensors 格式（推荐）
    NPCACHE = "npcache"  # NumPy 缓存
    DUMMY = "dummy"  # 虚拟权重（性能测试用）
    SHARDED_STATE = "sharded_state"  # 分片状态
    GGUF = "gguf"  # llama.cpp 量化格式
    BITSANDBYTES = "bitsandbytes"  # BitsAndBytes 量化
    MISTRAL = "mistral"  # Mistral 格式
    LAYERED = "layered"  # 分层加载
    FLASH_RL = "flash_rl"  # RL 训练量化模型 # For RL training with quantized models
    JAX = "jax"  # JAX 格式
    REMOTE = "remote"  # 远程加载
    REMOTE_INSTANCE = "remote_instance"  # 远程实例
    RDMA = "rdma"  # RDMA 传输
    LOCAL_CACHED = "local_cached"  # 本地缓存
    FASTSAFETENSORS = "fastsafetensors"  # 快速 SafeTensors
    PRIVATE = "private"  # 私有格式


@dataclass
class LoadConfig:
    """
    download_dir: Directory to download and load the weights, default to the
        default cache directory of huggingface.
    load_format: The format of the model weights to load:
        "auto" will try to load the weights in the safetensors format and
            fall back to the pytorch bin format if safetensors format is
            not available.
        "pt" will load the weights in the pytorch bin format.
        "safetensors" will load the weights in the safetensors format.
        "npcache" will load the weights in pytorch format and store
            a numpy cache to speed up the loading.
        "dummy" will initialize the weights with random values, which is
            mainly for profiling.
        "bitsandbytes" will load nf4 type weights.
        "flash_rl" will load weights with support for RL training
            with quantized models, enabling efficient weight reloading.
    ignore_patterns: The list of patterns to ignore when loading the model.
        Default to "original/**/*" to avoid repeated loading of llama's
        checkpoints.
    decryption_key_file: If set, decrypts the output files with a password read
        from this file (after PBKDF2).
    decrypt_max_concurrency: The maximum number of concurrent processes to decrypt the safetensor files. -1 means no limit.

    # ModelOpt-specific loading options
    modelopt_checkpoint_restore_path: Optional[str] = None
    modelopt_checkpoint_save_path: Optional[str] = None
    modelopt_export_path: Optional[str] = None
    """

    load_format: Union[str, LoadFormat] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(default_factory=dict)
    ignore_patterns: Optional[Union[List[str], str]] = None
    decryption_key_file: Optional[str] = None
    decrypt_max_concurrency: int = -1
    tp_rank: Optional[int] = None
    remote_instance_weight_loader_seed_instance_ip: Optional[str] = None
    remote_instance_weight_loader_seed_instance_service_port: Optional[int] = None
    remote_instance_weight_loader_send_weights_group_ports: Optional[List[int]] = None
    remote_instance_weight_loader_backend: Optional[str] = None
    remote_instance_weight_loader_transfer_engine: Optional[Any] = None

    # ModelOpt-specific loading options
    modelopt_checkpoint_restore_path: Optional[str] = None
    modelopt_checkpoint_save_path: Optional[str] = None
    modelopt_export_path: Optional[str] = None

    # ModelOpt configuration object
    modelopt_config: Optional[ModelOptConfig] = None

    # QuantizedRL-specific options (for FlashRL-style quantization)
    rl_quant_profile: Optional[str] = (
        None  # Path to rollout quantization profile (e.g., /root/profile.7b.pt)
    )

    # For multi-layer MTP
    draft_model_idx: Optional[int] = None

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = orjson.loads(model_loader_extra_config)
        self._verify_load_format()

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info(
                "Ignoring the following patterns when downloading weights: %s",
                self.ignore_patterns,
            )
        else:
            self.ignore_patterns = ["original/**/*"]

        # Create ModelOptConfig if not provided
        if self.modelopt_config is None:
            self.modelopt_config = ModelOptConfig(
                checkpoint_restore_path=self.modelopt_checkpoint_restore_path,
                checkpoint_save_path=self.modelopt_checkpoint_save_path,
                export_path=self.modelopt_export_path,
            )

    def _verify_load_format(self) -> None:
        if not isinstance(self.load_format, str):
            return

        load_format = self.load_format.lower()
        self.load_format = LoadFormat(load_format)

        rocm_not_supported_load_format: List[str] = []
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f
                for f in LoadFormat.__members__
                if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(
                f"load format '{load_format}' is not supported in ROCm. "
                f"Supported load formats are "
                f"{rocm_supported_load_format}"
            )
