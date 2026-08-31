import jax.numpy as jnp
from jax import tree_util
from jax.typing import ArrayLike

from .activation import Activation


@tree_util.register_pytree_node_class
class ReLU(Activation):
    def forward(self, x: ArrayLike) -> ArrayLike:
        return jnp.maximum(0, x)
