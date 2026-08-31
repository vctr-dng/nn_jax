import jax.numpy as jnp

from jax import Array

from .layer import Layer

class Flatten(Layer):

    def __init__(self, in_size: tuple[int, ...], key: Array | None = None, weights: Array | None = None, bias: Array | None = None):
        out_size: int = jnp.prod(jnp.array(in_size))
        super().__init__(in_size, out_size, key, weights, bias)

    def forward(self, x: Array) -> Array:
        pass