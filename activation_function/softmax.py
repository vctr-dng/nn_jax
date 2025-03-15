import jax.numpy as jnp
from jax.typing import ArrayLike

from activation_function import ActivationFunction


class Softmax(ActivationFunction):
    def forward(self, inputs: ArrayLike) -> ArrayLike:
        exps = jnp.exp(inputs - jnp.max(inputs))
        return exps / jnp.sum(exps, axis=1, keepdims=True)
