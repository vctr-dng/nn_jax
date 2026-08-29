from abc import abstractmethod

from jax.typing import ArrayLike

from nn_jax.module import Module


class Activation(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        self.inputs = inputs

    @abstractmethod
    def derivative(self, inputs: ArrayLike) -> ArrayLike:
        pass

    def backward(self, out_grad: ArrayLike) -> ArrayLike:
        return out_grad * self.derivative(self.inputs)

    def zero_grad(self):
        pass

    def _tree_flatten(self):
        return (), {}

    @classmethod
    def _tree_unflatten(cls, static_values, dynamic_values):
        return cls()
