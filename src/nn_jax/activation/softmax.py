import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .activation import Activation


class Softmax(Activation):
    def forward(self, inputs: ArrayLike) -> Array:
        super().forward(inputs)
        exps = jnp.exp(inputs - jnp.max(inputs, axis=1, keepdims=True))
        return exps / jnp.sum(exps, axis=1, keepdims=True)
