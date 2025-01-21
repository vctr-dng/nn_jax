import jax.numpy as jnp
from jax.typing import ArrayLike


class ReLU:
    def forward(self, inputs: ArrayLike) -> ArrayLike:
        return jnp.maximum(0, inputs)
