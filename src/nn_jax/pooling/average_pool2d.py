import jax.numpy as jnp

from jax import Array, tree_util

from .pooling import Pooling
from ..utils.im2col import extract_patches

@tree_util.register_pytree_node_class
class AvgPool2D(Pooling):
    
    def forward(self, x: Array) -> Array:
        patches = extract_patches(x, self.pool_size, self.stride)

        return jnp.mean(patches, axis=(2, 3))
