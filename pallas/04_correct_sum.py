import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
import numpy as np


def correct_sum_kernel(x_ref, o_ref):
    # 一个带有条件的装饰器：也是根据 grid 而言的 (Block_H, Block_W, Reduction_Size)
    # 第2维： Reduction_Size = 0
    @pl.when(pl.program_id(2) == 0)
    def _():
        # 把 SRAM 里的脏数据擦掉，初始化为 0
        o_ref[...] = jnp.zeros_like(o_ref)

    o_ref[...] += x_ref[...]


"""
(i,j,0) -> (i,j,1) -> (i,j,2) -> ... -> (i,j,7) 
0  1    ->    2    ->    3    -> ... ->    7  
"""


def correct_sum(x: jax.Array, block_size: tuple[int, ...] = (256, 256)) -> jax.Array:
    reduction_size, *out_shape = x.shape
    # We moved the reduction to the last axis of the grid.
    # (4, 4, 8)
    grid = (*(out // blk for out, blk in zip(out_shape, block_size)), reduction_size)
    return pl.pallas_call(
        correct_sum_kernel,
        grid=grid,
        # None in `block_shape` means we pick a size of 1 and squeeze it away
        in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (k, i, j))],
        out_specs=pl.BlockSpec(block_size, lambda i, j, k: (i, j)),
        out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
    )(x)

    """
    关键点1: 
    写成 None, 表示 直接取一个 (256,256),在搬运过程中将 k维度去除
    写成 1   , 表示 取 (1, 256, 256)

    关键点2: lambda 左边是针对 自己设置的grid而言的,右边才是实际的取的
    lambda i, j, k: (k, i, j)
    #      ^           ^
    #      |           |
    #   Grid给的     HBM要的
    """


x = jnp.ones((8, 1024, 1024))
result = correct_sum(x)
print(result)

"""
总结 SRAM 的三个反直觉特性：
它不自动清零： 就像公共厕所，进去如果不先检查，可能会看到上一个人留下的东西。
（所以必须手动 if step==0: o_ref[...] = 0）

它是唯一的桥梁： 所有计算必须在它上面发生。
它的数据寿命由 Grid 决定：

Grid 变动了（切片变了） -> 数据被冲刷(写回HBM)。
Grid 没变  （切片没变） -> 数据驻留 (仍在SRAM)。
"""
