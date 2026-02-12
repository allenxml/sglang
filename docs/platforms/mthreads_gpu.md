# Moore Threads GPUs

This document describes how run SGLang on Moore Threads GPUs. If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

**中文对照**：# Moore Threads GPU

本文档介绍如何在 Moore Threads GPU 上运行 SGLang。如果您遇到问题或有疑问，请[提交 issue](https://github.com/sgl-project/sglang/issues)。

## Install SGLang

You can install SGLang using one of the methods below.

**中文对照**：## 安装 SGLang

您可以使用以下方法之一安装 SGLang。

### Install from Source

```bash
# Use the default branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_musa.py install

# Install sglang python package
cd ..
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```

**中文对照**：### 从源码安装

```bash
# 使用默认分支
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 编译 sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_musa.py install

# 安装 sglang python 包
cd ..
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `sgl-kernel/setup_musa.py` | Moore Threads GPU 的 MUSA 专用内核编译设置 |
| `python/sglang/srt/layers/utils/multi_platform.py` | 平台抽象：MUSA 设备检测和分发 |

### 集成要点

- **安装**：使用 `pyproject_other.toml` 和 `pip install -e "python[all_musa]"` 安装 MUSA 依赖
- **内核编译**：`sgl-kernel` 通过 `python setup_musa.py install` 编译（需要 MUSA 工具链）
