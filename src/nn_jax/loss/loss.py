from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


class Loss(ABC):
    def __call__(self, y_pred: ArrayLike, y_true: ArrayLike) -> Array:
        calculated_loss: Array = self.calculate_loss(y_pred, y_true)
        loss = jnp.mean(calculated_loss)
        return loss

    @abstractmethod
    def calculate_loss(self, y_pred: ArrayLike, y_true: ArrayLike) -> Array:
        raise NotImplementedError("Subclasses must implement calculate_loss method.")
