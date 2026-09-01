import jax.numpy as jnp

from jax import Array, tree_util

from .pooling import Pooling
from ..utils.im2col import extract_patches

@tree_util.register_pytree_node_class
class MaxPool2D(Pooling):
    def __init__(self, in_size: tuple[int, ...], pool_size: tuple[int, ...], stride: int = 1):
        super().__init__(in_size, pool_size, stride)

    def forward(self, x: Array) -> Array:
        patches = extract_patches(x, self.pool_size, self.stride)

        return jnp.max(patches, axis=(2, 3))
