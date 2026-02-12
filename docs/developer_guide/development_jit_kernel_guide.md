# Development Guide for JIT Kernels

## Environment Setup

We strongly recommend using `clangd` as the language server for JIT kernel development.
For Ubuntu/Debian, you can download clangd from [apt.llvm.org](https://apt.llvm.org/).
If you are using VS Code, we recommend installing the `clangd` extension for better IDE integration.

**中文对照**：## 环境设置

我们强烈建议使用 `clangd` 作为 JIT 内核开发的语言服务器。对于 Ubuntu/Debian，你可以从 [apt.llvm.org](https://apt.llvm.org/) 下载 clangd。如果你使用 VS Code，我们建议安装 `clangd` 扩展以获得更好的 IDE 集成。

All JIT-related files are located in `python/sglang/jit_kernel`.
Unlike `sgl-kernel`, which compiles CUDA/C++ binaries ahead of time (AOT), just-in-time (JIT) kernels are compiled at runtime.
Consequently, a static `compile_commands.json` cannot be generated.
To enable code completion with `clangd`, run `python -m sglang.jit_kernel` to generate a `.clangd` configuration file in your current directory.
After generating the file, restart the clangd language server. It should now recognize all JIT kernel files.

**中文对照**：所有 JIT 相关文件位于 `python/sglang/jit_kernel`。
与提前编译 CUDA/C++ 二进制文件的 `sgl-kernel` 不同，just-in-time (JIT) 内核在运行时编译。
因此，无法生成静态 `compile_commands.json`。
要启用 `clangd` 的代码补全功能，运行 `python -m sglang.jit_kernel` 在当前目录生成 `.clangd` 配置文件。
生成文件后，重启 clangd 语言服务器。它现在应该能够识别所有 JIT 内核文件。

## Code Structure

**中文对照**：## 代码结构

### C++ Implementation

C++ source code is located in `python/sglang/jit_kernel/csrc`.
Reusable functions should be placed in `python/sglang/jit_kernel/include`.

**中文对照**：### C++ 实现

C++ 源代码位于 `python/sglang/jit_kernel/csrc`。
可重用函数应放置在 `python/sglang/jit_kernel/include` 中。

We use [tvm-ffi](https://github.com/apache/tvm-ffi) for efficient foreign language bindings.
Refer to the [documentation](https://tvm.apache.org/ffi/) for advanced usage, such as exporting C++ objects.
Typically, `tvm::ffi::TensorView` is sufficient for passing PyTorch Tensors from Python.

**中文对照**：我们使用 [tvm-ffi](https://github.com/apache/tvm-ffi) 来实现高效的外语绑定。
有关高级用法（例如导出 C++ 对象），请参阅[文档](https://tvm.apache.org/ffi/)。
通常，`tvm::ffi::TensorView` 对于从 Python 传递 PyTorch 张量已经足够。

### Python Interface

**中文对照**：### Python 接口

Python interfaces are defined in `python/sglang/jit_kernel`.
The `load_jit` utility function in `python/sglang/jit_kernel/utils.py` loads and returns the compiled module.
To export a C++ function (e.g., `cpp_func`), pass `cuda_wrappers=[("func", "cpp_func")]` to `load_jit`.
The function can then be called in Python as `module.func`.

**中文对照**：Python 接口定义在 `python/sglang/jit_kernel` 中。
`python/sglang/jit_kernel/utils.py` 中的 `load_jit` 实用函数加载并返回编译后的模块。
要导出 C++ 函数（例如 `cpp_func`），将 `cuda_wrappers=[("func", "cpp_func")]` 传递给 `load_jit`。
然后可以在 Python 中调用该函数为 `module.func`。

For caching compiled modules, prefer `sglang.jit_kernel.utils.cache_once` over `functools.lru_cache`.
`functools.lru_cache` is not compatible with `torch.compile`.

**中文对照**：对于缓存编译后的模块，优先使用 `sglang.jit_kernel.utils.cache_once` 而不是 `functools.lru_cache`。
`functools.lru_cache` 与 `torch.compile` 不兼容。

### C++ Utilities

The following C++ utilities are available:

**中文对照**：### C++ 实用工具

以下 C++ 实用工具可用：

#### Integer Range

Similar to PyTorch, we provide an `irange` function to represent an integer range.

```C++
#include <sgl_kernel/utils.h>

void test() {
  for (auto i : host::irange(100)) { // [0, 100)
    // do something
  }
  for (auto i : host::irange(0, 100)) { // [0, 100)
    // do something
  }
}

```

**中文对照**：#### 整数范围

与 PyTorch 类似，我们提供 `irange` 函数来表示整数范围。

#### Runtime Checking

`RuntimeCheck` validates conditions at runtime. It accepts optional arguments for error reporting.
If the check fails, these arguments are output to aid debugging.
`RuntimeDeviceCheck` verifies the status of the last kernel launch.

```C++
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

void test() {
  host::RuntimeCheck(1 + 1 == 2, 1 + 1, " != ", 2);
  host::RuntimeDeviceCheck();
  // check the provided `cudaError_t`
  host::RuntimeDeviceCheck(cudaGetLastError());
}

```

**中文对照**：#### 运行时检查

`RuntimeCheck` 在运行时验证条件。它接受可选参数用于错误报告。
如果检查失败，这些参数将输出以帮助调试。
`RuntimeDeviceCheck` 验证最后一次内核启动的状态。

#### Tensor Checking

**中文对照**：#### 张量检查

#### Tensor Checking

`TensorMatcher` provides a readable way to validate and extract tensor shape information.

```cpp
#include <sgl_kernel/tensor.h>

void test(const tvm::ffi::TensorView k_cache, const tvm::ffi::TensorView v_cache) {
  using namespace host;

  auto D = SymbolicSize{"D"};  // cache dimension
  auto N = SymbolicSize{"N"};  // kvcache stride
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};

  TensorMatcher({-1, D})  //
      .with_strides({N, 1})
      .with_dtype<int32_t, int64_t>(dtype)
      .with_device<kDLCUDA, kDLCPU>(device)
      .verify(k_cache)
      .verify(v_cache);
}
```

Configure the `TensorMatcher` with expected stride, dtype, and device properties before verification.
- If `with_strides` is omitted, the tensor is expected to be contiguous.
- Template arguments in `with_dtype` restrict the allowed data types.
- Template arguments in `with_device` restrict the allowed devices.
- Values passed to `with_xxx` methods enforce equality checks.
- Passing `-1` for size or stride allows matching any value.

A `Symbolic` variable must resolve to the same value across all verifications.
Use `.unwrap()` to retrieve the matched value after verification.

> Note: `TensorMatcher` is a temporary expression and should not be stored in a variable.

> Tip: Add `//` at the end of the `TensorMatcher` chain to enforce proper indentation.

**中文对照**：在验证之前，使用预期的步幅、dtype 和设备属性配置 `TensorMatcher`。
- 如果省略 `with_strides`，则期望张量是连续的。
- `with_dtype` 中的模板参数限制允许的数据类型。
- `with_device` 中的模板参数限制允许的设备。
- 传递给 `with_xxx` 方法的值强制执行相等性检查。
- 对于大小或步幅传递 `-1` 允许匹配任何值。

`Symbolic` 变量必须在所有验证中解析为相同的值。
使用 `.unwrap()` 在验证后检索匹配的值。

> 注意：`TensorMatcher` 是一个临时表达式，不应存储在变量中。

> 提示：在 `TensorMatcher` 链末尾添加 `//` 以强制执行正确的缩进。

#### Kernel Launching

`LaunchKernel::resolve_device` retrieves the current `cudaStream` from PyTorch.
Kernels can also be launched directly using `LaunchKernel`.

```cpp
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>

__global__ void kernel() {}

void test() {
  const auto num_blocks = 1;
  const auto num_threads = 32;
  const auto dynamic_smem = 0;

  DLDevice dev;  // suppose this is initialized properly
  host::LaunchKernel(num_blocks, num_threads, dev)(kernel);

  cudaStream_t stream = host::LaunchKernel::resolve_device(dev);
  host::LaunchKernel(num_blocks, num_threads, stream, dynamic_smem)(kernel);
}

```

**中文对照**：#### 内核启动

`LaunchKernel::resolve_device` 从 PyTorch 检索当前的 `cudaStream`。
内核也可以使用 `LaunchKernel` 直接启动。

## Add new kernels

This section walks through a complete, end-to-end example of adding a new JIT kernel to the system.
We use a simple add_constant kernel as a running example, which adds a constant integer value to every element of an input tensor.

Conceptually, the Python interface looks like this:

```python
def add_constant(src: torch.Tensor, c: int):
    return src + c
```

### STEP 1: Write the C++ kernel

Write your CUDA kernel in [jit_kernel/csrc/add_constant.cuh](../../python/sglang/jit_kernel/csrc/add_constant.cuh). For demonstration purposes, we pass the constant value as a template parameter.

```cpp
#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/utils.h>    // For div_ceil, RuntimeCheck

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <int32_t kConstant>
__global__ void add_constant_kernel(int32_t* dst, const int32_t* src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx] + kConstant;
  }
}

constexpr size_t kBlockSize = 256;

// You can also use struct with static method as an alternative
template <int32_t kConstant>
void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;

  // 1. Validate input tensors
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  TensorMatcher({N})                  // 1D tensor, must be contiguous
      .with_dtype<int32_t>()          // must be int32
      .with_device<kDLCUDA>(device_)  // must be on CUDA device
      .verify(dst)                    // check tensor dst
      .verify(src);                   // check tensor src

  // 2. Extract required parameters, prepare for kernel launch
  const size_t num_elements = N.unwrap();
  const size_t grid_size = div_ceil(num_elements, kBlockSize);
  const DLDevice device = device_.unwrap();
  // some extra runtime checks using host::RuntimeCheck
  RuntimeCheck(num_elements > 0, "We only support non-empty tensors, got num_elements = ", num_elements);

  // 3. Launch the kernel. Error code will be automatically checked.
  LaunchKernel(grid_size, kBlockSize, device /*, dynamic_smem*/)(
      // kernel function
      add_constant_kernel<kConstant>,
      // kernel arguments
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      num_elements);
}

}  // namespace

```

### STEP 2: Create Python Interfaces

Next, expose the kernel through a Python wrapper.
Create a new file at [jit_kernel/add_constant.py](../../python/sglang/jit_kernel/add_constant.py) and expose the needed interfaces.

```python
from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_add_constant_module(constant: int) -> Module:
    args = make_cpp_args(constant)  # pass all the template argument
    return load_jit(
        "add_constant",
        *args,
        cuda_files=["add_constant.cuh"],
        cuda_wrappers=[("add_constant", f"add_constant<{args}>")],
    )


def add_constant(src: torch.Tensor, constant: int) -> torch.Tensor:
    dst = torch.empty_like(src)
    module = _jit_add_constant_module(constant)
    module.add_constant(dst, src)
    return dst

```

### STEP 3: Use your kernel

Finally, import and use the kernel like a regular Python function:

```python
from sglang.jit_kernel.add_constant import add_constant
```

For a complete, runnable example, refer to [test_add_constant.py](../../python/sglang/jit_kernel/tests/test_add_constant.py).

## 代码实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `python/sglang/jit_kernel/` | JIT 内核根目录：Python 接口 + 构建系统 |
| `python/sglang/jit_kernel/csrc/` | C++/CUDA 内核源文件（`.cu`、`.cuh`） |
| `python/sglang/jit_kernel/include/` | 可重用 C++ 头文件（`sgl_kernel/utils.h`、`sgl_kernel/tensor.h`、`sgl_kernel/utils.cuh`） |
| `python/sglang/jit_kernel/utils.py` | `load_jit()` 构建工具、`cache_once` 装饰器（torch.compile 兼容）、`make_cpp_args` |

### 关键代码逻辑

- **JIT 编译**：`load_jit()` 使用 tvm-ffi 在运行时编译 CUDA/C++ 文件；返回包含可调用 Python 函数的模块
- **张量验证**：`TensorMatcher` 配合 `SymbolicSize`/`SymbolicDType`/`SymbolicDevice` 进行声明式张量形状/类型检查
- **内核启动**：`LaunchKernel(grid, block, device)` 解析 PyTorch CUDA 流并启动，自动错误检查
- **缓存**：`cache_once` 装饰器（而非 `lru_cache`）确保编译模块的 torch.compile 兼容性

### 集成要点

- **IDE 设置**：运行 `python -m sglang.jit_kernel` 生成 `.clangd` 配置以启用代码补全
- **添加内核**：3 步流程 — 在 `csrc/` 中编写 `.cuh` → 使用 `load_jit()` 创建 Python 包装器 → 导入并使用
- **vs sgl-kernel**：JIT 内核在运行时编译；`sgl-kernel` 提前编译（AOT）并作为独立 PyPI 包发布
