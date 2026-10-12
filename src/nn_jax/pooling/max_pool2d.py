from jax import Array, tree_util

from ..utils.im2col import extract_patches
from .pooling import Pooling


@tree_util.register_pytree_node_class
class MaxPool2D(Pooling):
    def forward(self, x: Array) -> Array:
        patches = extract_patches(x, self.pool_size, self.stride)

        return patches.max(axis=(2, 3))
