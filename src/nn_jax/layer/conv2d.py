import jax.numpy as jnp

from jax import Array

from .layer import Layer

class Conv2D(Layer):
    def __init__(
        self,
        in_channels: tuple[int, ...],
        out_channels: tuple[int, ...],
        kernel_size: tuple[int, ...],
        stride: int = 1,
        padding: int = 0,
        key: Array | None = None,
        weights: Array | None = None,
        bias: Array | None = None,
    ):
        super().__init__(in_channels, out_channels, key, weights, bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Array) -> Array:
        pass