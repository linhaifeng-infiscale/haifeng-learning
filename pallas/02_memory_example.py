import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
import numpy as np

# Note: This is a TPU example.


# 内核函数
def add_matrices_kernel(x_sram_ref, y_sram_ref, z_sram_ref):
    # 1. Load: 从 SRAM 读取到 寄存器
    # x_sram_ref 是指向 SRAM 中数据的引用 (Ref)。
    # 当我们对其进行切片读取 [:, :] 时，数据被加载到了计算单元的寄存器中。
    x_regs = x_sram_ref[:, :]
    y_regs = y_sram_ref[:, :]

    # 2. Compute: 在 寄存器 中执行向量加法
    # 这一步极其快速，利用了 TPU 的向量处理单元 (VPU)。
    z_regs = x_regs + y_regs

    # 3. Store: 从 寄存器 写回 SRAM
    # 计算结果必须存回 SRAM，才能最终被搬运回 HBM。
    z_sram_ref[:, :] = z_regs


# 驱动函数
def add_matrices(x: jax.Array, y: jax.Array) -> jax.Array:
    # pl.pallas_call 是指挥官。
    # 在这个简单的例子中，它默默完成了所有繁重的工作：
    # 1. Alloc: 在 TPU 的 SRAM 中分配 x, y, z 的空间。
    # 2. Copy In: 将 x, y 从 HBM (DDR) 复制到 SRAM。
    # 3. Execute: 启动 add_matrices_kernel。
    # 4. Copy Out: 将结果 z 从 SRAM 复制回 HBM。
    z = pl.pallas_call(
        add_matrices_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)
    return z


x, y = jnp.ones((512, 512)), jnp.ones((512, 512))
print(add_matrices(x, y))

"""
这段代码虽然清晰，但在高性能计算（HPC）场景下是不可用的，因为它撞上了“内存墙”。

瓶颈一：内存容量 (Memory Capacity)

现象： SRAM (VMEM) 非常小。如文中提到，TPU v5p 的 VMEM 只有 96MB，而普通的 GPU L1/L2 Cache 更小。
问题： pallas_call 在这里尝试将整个数组一次性塞进 SRAM。
512x512 的 float32 数组只有 1MB，没问题。
但在大模型训练中，矩阵往往是 [8192, 8192] 甚至更大。
后果： 如果输入数组太大，超过了 SRAM 大小，程序会直接报错（OOM）或无法运行。

瓶颈二：内存带宽 (Memory Bandwidth)
现象： HBM 与 SRAM 之间的搬运速度（Copy）远慢于寄存器上的加法运算（Compute）。
算术强度 (Arithmetic Intensity)： 这是一个典型的 IO Bound (IO受限) 算子。
我们搬运了 2 个数进来，只做了 1 次加法，又搬运了 1 个数出去。
计算几乎瞬间完成，绝大部分时间 TPU 都在“干瞪眼”，等待数据从 HBM 慢慢爬进 SRAM。
后果： 你的 TPU 算力利用率（FLOPS Utilization）可能连 1% 都不到。

"""
