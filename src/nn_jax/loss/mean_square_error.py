import jax.numpy as jnp
from jax.typing import ArrayLike

from .loss import Loss


class MeanSquareError(Loss):
    def calculate_loss(self, pred: ArrayLike, target: ArrayLike) -> ArrayLike:
        return jnp.mean((pred - target) ** 2)
