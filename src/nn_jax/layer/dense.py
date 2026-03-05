import jax.numpy as jnp
import jax.random as random

from jax import Array, tree_util
from jax.typing import ArrayLike

from .layer import Layer

SEED = 0
key = random.key(SEED)


class Dense(Layer):
    weight_type = jnp.float32

    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.weights: Array = 0.1 * random.normal(
            key, shape=(self.output_size, self.input_size), dtype=self.weight_type
        )
        self.biases: ArrayLike = jnp.zeros(self.output_size, dtype=self.weight_type)
        self.output: Array = self.forward(jnp.zeros(self.input_size))

    def forward(self, input: ArrayLike) -> Array:
        self.output = self.weights @ input + self.biases
        return self.output

    def _tree_flatten(self):
        dynamic_values = (
            self.weights,
            self.biases,
            self.output,
        )
        static_values = {
            "input_size": self.input_size,
            "output_size": self.output_size,
        }
        return (dynamic_values, static_values)

    @classmethod
    def _tree_unflatten(cls, static_values, dynamic_values):
        dense = cls(
            static_values["input_size"],
            static_values["output_size"],
        )
        dense.weights = dynamic_values[0]
        dense.biases = dynamic_values[1]
        dense.output = dynamic_values[2]
        return dense


tree_util.register_pytree_node(
    Dense,
    Dense._tree_flatten,
    Dense._tree_unflatten,
)
