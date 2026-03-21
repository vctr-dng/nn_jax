from abc import abstractmethod

from jax import Array
from jax.typing import ArrayLike


class Module:
    @abstractmethod
    def __init__(self):
        self.inputs: ArrayLike
        self.outputs: ArrayLike

    @abstractmethod
    def forward(self, inputs: ArrayLike) -> ArrayLike:
        pass

    @abstractmethod
    def backward(self, out_grad: ArrayLike) -> ArrayLike:
        pass

    @property
    def parameters(self) -> list[Array]:
        return []

    @parameters.setter
    def parameters(self, new_parameters: list[Array]):
        pass

    @property
    def gradients(self) -> list[Array]:
        return []

    @abstractmethod
    def zero_grad(self):
        pass
