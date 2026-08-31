from abc import abstractmethod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from nn_jax.module import Module


class Layer(Module):
    def __init__(self, in_size: ArrayLike, out_size: ArrayLike):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.inputs = jnp.zeros((self.in_size, 1))
        self.outputs = jnp.zeros((self.out_size, 1))

    def forward(self, inputs: Array) -> Array:
        self.inputs = inputs

    @abstractmethod
    def backward(self, out_grad: ArrayLike) -> Array:
        pass
