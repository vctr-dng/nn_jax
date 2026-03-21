import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .activation import Activation


class Tanh(Activation):
    def forward(self, inputs: ArrayLike) -> Array:
        super().forward(inputs)
        return jnp.tanh(inputs)

    def derivative(self, inputs: ArrayLike) -> Array:
        return 1 - jnp.tanh(inputs) ** 2
