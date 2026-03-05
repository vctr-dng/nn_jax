import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .activation_function import ActivationFunction


class ReLU(ActivationFunction):
    def forward(self, inputs: ArrayLike) -> Array:
        return jnp.maximum(0, inputs)
