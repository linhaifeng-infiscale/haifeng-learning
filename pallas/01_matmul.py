from functools import partial

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np


def matmul_kernel(x_ref, y_ref, z_ref, *, activation):
    z_ref[...] = activation(x_ref[...] @ y_ref[...])


def matmul(x: jax.Array, y: jax.Array, *, activation):
    return pl.pallas_call(
        partial(matmul_kernel, activation=activation),
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((x.shape[0] // 2, x.shape[1]), lambda i, j: (i, 0)),
            pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((x.shape[0] // 2, y.shape[1] // 2), lambda i, j: (i, j)),
    )(x, y)


# k1, k2 = jax.random.split(jax.random.key(0))
# x = jax.random.normal(k1, (1024, 1024))
# y = jax.random.normal(k2, (1024, 1024))
# z = matmul(x, y, activation=jax.nn.relu)

k1, k2 = jax.random.split(jax.random.key(0))
# 加入了 batch 维度
x = jax.random.normal(k1, (4, 1024, 1024))
y = jax.random.normal(k2, (4, 1024, 1024))
# JAX 的变换函数（如 vmap）通常只希望处理数据参数（张量），而不希望处理配置参数（如激活函数、步长、布尔开关）。
# vmap:单卡批处理  pmap:多卡并行
z = jax.vmap(partial(matmul, activation=jax.nn.relu))(x, y)
"""
why not lambda?
vmap(lambda x, y: matmul(x, y, activation=jax.nn.relu))
因为 lambda 表达式是匿名的，无法被 JAX 的 JIT 编译器识别和优化。
使用 partial 可以明确地传递函数及其参数，使得 JAX 能够正确地处理和编译这些函数调用，从而提高性能。
"""
np.testing.assert_allclose(z, jax.nn.relu(jax.vmap(jnp.matmul)(x, y)))
