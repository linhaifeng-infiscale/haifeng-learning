from functools import partial

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np


def add_vectors_kernel(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y


@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        add_vectors_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)


# result = add_vectors(jnp.arange(8), jnp.arange(8))
# print(result)  # [0 2 4 6 8 10 12 14]


@jax.jit
def add_sliced_kernel(x_ref, y_ref, o_ref):
    small_mid = x_ref.shape[0] // 2

    x_left = x_ref.at[:small_mid]
    x_right = x_ref.at[small_mid:]
    y_left = y_ref.at[:small_mid]
    y_right = y_ref.at[small_mid:]

    # The output shape is (4*small_mid).
    large_mid = 2 * small_mid
    o_ref.at[:large_mid][:small_mid] = x_left[...] + y_left[...]
    o_ref.at[:large_mid][small_mid:] = x_left[...] + y_right[...]
    o_ref.at[large_mid:][:small_mid] = x_right[...] + y_left[...]
    o_ref.at[large_mid:][small_mid:] = x_right[...] + y_right[...]


def add_sliced(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        add_sliced_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0] * 2,), x.dtype),
    )(x, y)


x = jnp.arange(1024)
y = jnp.arange(1024)
result = add_sliced(x, y)
print(result)
"""
TPU v6e 上写 Kernel,处理的数据粒度必须更粗,太碎小的切片会导致编译器无法生成流水线指令.

硬件的拒绝：TPU 的 地址生成单元 (Address Generation Unit) 非常“硬”。
它规定：“如果你想在一个大内存里切一个小窗口，这个窗口的起始位置和长度必须对齐到我的向量宽度（512）。”

Ref (引用) 指向的是 VMEM (Vector Memory)。
在 VMEM 上做切片（.at）受限于 VMEM 的寻址硬件限制（必须对齐）。

add_vectors 没有进行手动内存切片 (.at[...])，而是全量加载。
编译器可以对全量加载进行自动填充（Padding）或掩码（Masking）处理，但它无法替你的手动切片逻辑做决定。
"""
