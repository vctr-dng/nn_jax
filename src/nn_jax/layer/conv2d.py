from jax import Array, random

from .layer import Layer


class Conv2D(Layer):
    """
    x follows (H, W, C) format
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        key: Array | None = None,
        weights: Array | None = None,
        bias: Array | None = None,
        stride: int = 1,
        padding: int = 0,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super().__init__(in_channels, out_channels, key, weights, bias)

    def _init_weights(self, key: Array):
        self.weights = 0.1 * random.normal(
            key, (*self.kernel_size, self.in_size, self.out_size)
        )

    @property
    def _weights_shape(self) -> tuple[int, ...]:
        return (*self.kernel_size, self.in_size, self.out_size)

    def forward(self, x: Array) -> Array:
        pass
