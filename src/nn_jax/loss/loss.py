from abc import ABC, abstractmethod

from jax.typing import ArrayLike


class Loss(ABC):
    def __call__(self, pred: ArrayLike, target: ArrayLike) -> ArrayLike:
        loss: ArrayLike = self.calculate_loss(pred, target)
        return loss

    @abstractmethod
    def calculate_loss(self, pred: ArrayLike, target: ArrayLike) -> ArrayLike:
        pass
