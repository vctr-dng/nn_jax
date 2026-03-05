from abc import ABC, abstractmethod

from jax.typing import ArrayLike


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, inputs: ArrayLike) -> ArrayLike:
        pass

    def _tree_flatten(self):
        return (), {}

    @classmethod
    def _tree_unflatten(cls, static_values, dynamic_values):
        return cls()
