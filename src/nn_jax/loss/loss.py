from abc import ABC, abstractmethod

from jax.typing import ArrayLike


class Loss(ABC):
    def __call__(self, y_pred: ArrayLike, y_true: ArrayLike) -> ArrayLike:
        loss: ArrayLike = self.calculate_loss(y_pred, y_true)
        return loss

    @abstractmethod
    def calculate_loss(self, y_pred: ArrayLike, y_true: ArrayLike) -> ArrayLike:
        pass

    @abstractmethod
    def derivative(self, y_pred: ArrayLike, y_true: ArrayLike) -> ArrayLike:
        pass
