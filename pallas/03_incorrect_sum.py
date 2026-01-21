import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
import numpy as np


# Note: This is a TPU example.


# Warning: this implementation is incorrect!
def incorrect_sum_kernel(x_ref, o_ref):
    o_ref[...] += x_ref[...]


def incorrect_sum(x: jax.Array, block_size: tuple[int, ...] = (256, 256)) -> jax.Array:
    reduction_size, *out_shape = x.shape
    grid = (reduction_size, *(out // blk for out, blk in zip(out_shape, block_size)))
    # print(grid) # [8, 4, 4]
    return pl.pallas_call(
        incorrect_sum_kernel,
        grid=grid,
        # None in `block_shape` means we pick a size of 1 and squeeze it away
        in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (i, j, k))],
        out_specs=pl.BlockSpec(block_size, lambda i, j, k: (j, k)),
        out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
    )(x)


x = jnp.ones((8, 1024, 1024))
result = incorrect_sum(x)
print(result)

"""
correct:
[[8., 8., 8., ..., 8., 8., 8.],
 [8., 8., 8., ..., 8., 8., 8.],
 [8., 8., 8., ..., 8., 8., 8.],
 ...,
 [8., 8., 8., ..., 8., 8., 8.],
 [8., 8., 8., ..., 8., 8., 8.],
 [8., 8., 8., ..., 8., 8., 8.]]

incorrect:
[[58. 58. 58. ... 59. 59. 59.]
 [58. 58. 58. ... 59. 59. 59.]
 [58. 58. 58. ... 59. 59. 59.]
 ...
 [64. 64. 64. ... 65. 65. 65.]
 [64. 64. 64. ... 65. 65. 65.]
 [64. 64. 64. ... 65. 65. 65.]]
"""

"""
为什么结果不对？

如果连续两次迭代的输出切片位置完全相同，流水线才不会清空 SRAM，让你在旧值上继续加。

在计算机循环中，通常最右边的维度变化最快（最内层循环），最左边的维度变化最慢（最外层循环）。 
所以，Pallas 会这样执行迭代：

Iter 0: i=0 (第1层), j=0, k=0。
动作： 计算第1层的数据，写入 Output Block (0,0)。

Iter 1: i=0, j=0, k=1 (注意：这里变的是 k，不是 i)。
问题来了： 我们的输出位置变了！从 Block (0,0) 变成了 Block (0,1)。

Pallas 的反应： “哎呀，老板换地方了。把刚才 Block (0,0) 的结果写回 HBM，清空 SRAM，准备接待 Block (0,1)。”
后果： Block (0,0) 的累加只进行了一次（只加了第1层），就被强制写回并遗忘了。
"""

# s = jnp.sum(x, axis=0)

# print(s)
