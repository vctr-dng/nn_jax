import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .activation import Activation


class ReLU(Activation):
    def forward(self, inputs: ArrayLike) -> Array:
        super().forward(inputs)
        return jnp.maximum(0, inputs)

    def derivative(self, inputs: ArrayLike) -> Array:
        return jnp.where(inputs > 0, 1, 0)
