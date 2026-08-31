from abc import ABC, abstractmethod

from jax.typing import ArrayLike


class Module(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x: ArrayLike) -> ArrayLike:
        pass
