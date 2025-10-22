import jax.numpy as jnp
from jax.typing import ArrayLike

from nn_jax.activation_function import ActivationFunction


class Softmax(ActivationFunction):
    def forward(self, inputs: ArrayLike) -> ArrayLike:
        exps = jnp.exp(inputs - jnp.max(inputs, axis=1, keepdims=True))
        return exps / jnp.sum(exps, axis=1, keepdims=True)
