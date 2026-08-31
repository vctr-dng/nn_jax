import jax.numpy as jnp
from jax import Array, tree_util
from jax.typing import ArrayLike

from .activation import Activation


@tree_util.register_pytree_node_class
class Softmax(Activation):
    def forward(self, x: ArrayLike) -> Array:
        exps = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
        return exps / jnp.sum(exps, axis=-1, keepdims=True)
