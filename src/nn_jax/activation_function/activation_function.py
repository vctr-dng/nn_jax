from abc import ABC, abstractmethod

from jax.typing import ArrayLike


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, inputs: ArrayLike) -> ArrayLike:
        pass
