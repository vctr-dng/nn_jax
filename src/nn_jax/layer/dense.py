from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax.random as random
from jax import Array, vmap

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
        self.biases: Array = jnp.zeros(self.output_size, dtype=self.weight_type)
        self.output: Array = self.forward(jnp.zeros((1, self.input_size)))

    def forward(self, inputs: Array) -> Array:
        partial_forward: Callable[[Array], Array] = partial(
            single_forward, self.weights, self.biases
        )

        # vmap transforms partial_forward to work on batches
        # in_axes=0: iterate over the first axis (batch dimension) of inputs
        # out_axes=0: stack results along the first axis (batch dimension)
        vectorized_forward = vmap(partial_forward, in_axes=0, out_axes=0)

        self.output = vectorized_forward(inputs)
        return self.output


def single_forward(weights: Array, biases: Array, x: Array) -> Array:
    return weights @ x + biases
