import jax.numpy as jnp
from jax.typing import ArrayLike

from nn_jax.activation_function import ActivationFunction


class ReLU(ActivationFunction):
    def forward(self, inputs: ArrayLike) -> ArrayLike:
        return jnp.maximum(0, inputs)
