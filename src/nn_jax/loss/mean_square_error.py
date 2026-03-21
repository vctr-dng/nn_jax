import jax.numpy as jnp
from jax.typing import ArrayLike

from .loss import Loss


class MeanSquareError(Loss):
    def calculate_loss(self, y_pred: ArrayLike, y_true: ArrayLike) -> ArrayLike:
        return jnp.mean((y_pred - y_true) ** 2)

    def derivative(self, y_pred: ArrayLike, y_true: ArrayLike) -> ArrayLike:
        return 2 * (y_pred - y_true) / y_true.size
