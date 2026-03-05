from abc import ABC, abstractmethod

from jax import Array
from jax.typing import ArrayLike


class Layer(ABC):
    def __init__(self, input_size: ArrayLike, output_size: ArrayLike):
        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def forward(self, inputs: Array) -> Array:
        pass
