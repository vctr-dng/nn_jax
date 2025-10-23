from functools import partial

import jax.numpy as jnp
import jax.random as random
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from .layer import Layer

SEED = 0
key = random.key(SEED)


class Dense(Layer):
    weight_type = jnp.float32

    def __init__(self, input_size: ArrayLike, output_size: ArrayLike):
        super().__init__(input_size, output_size)
        self.weights: Array = 0.1 * random.normal(
            key, shape=(self.output_size, self.input_size), dtype=self.weight_type
        )
        self.biases: Array = jnp.zeros((1, self.output_size), dtype=self.weight_type)
        self.output: ArrayLike = self.forward(jnp.zeros((1, self.input_size)))

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        partial_forward = partial(single_forward, self.weights, self.biases)
        vectorized_forward = jit(vmap(partial_forward, in_axes=0, out_axes=1))
        self.output = vectorized_forward(inputs)
        return self.output


def single_forward(weights: Array, biases: Array, x: Array) -> Array:
    return weights @ x + biases
