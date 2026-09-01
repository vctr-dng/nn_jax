import jax.numpy as jnp
from jax import Array, random, tree_util

from ..utils.im2col import extract_patches
from .layer import Layer


@tree_util.register_pytree_node_class
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
        self.kernel_size: tuple[int, ...] = kernel_size
        self.stride: int = stride
        self.padding: int = padding
        super().__init__(in_channels, out_channels, key, weights, bias)

    def _init_weights(self, key: Array):
        self.weights = 0.1 * random.normal(
            key, (*self.kernel_size, self.in_size, self.out_size)
        )

    @property
    def _weights_shape(self) -> tuple[int, ...]:
        return (*self.kernel_size, self.in_size, self.out_size)

    @classmethod
    def from_parameters(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        weights: Array,
        bias: Array,
        stride: int = 1,
        padding: int = 0,
    ) -> "Layer":
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            weights=weights,
            bias=bias,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: Array) -> Array:
        padded_x = jnp.pad(
            x,
            (
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0),
            ),
            mode="constant",
        )
        patches = extract_patches(padded_x, self.kernel_size, self.stride)

        H_out, W_out = patches.shape[:2]
        patches = patches.reshape(H_out * W_out, -1)
        weights = self.weights.reshape(-1, self.out_size)
        output = patches @ weights + self.bias

        return output.reshape(H_out, W_out, self.out_size)

    def tree_flatten(
        self,
    ) -> tuple[tuple[Array, Array], tuple[int, int, int, int, int]]:
        children = (
            self.weights,
            self.bias,
        )
        aux_data = (
            self.in_size,
            self.out_size,
            self.kernel_size,
            self.stride,
            self.padding,
        )

        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[int, int, tuple[int, ...], int, int],
        children: tuple[Array, Array],
    ) -> "Conv2D":
        in_size, out_size, kernel_size, stride, padding = aux_data
        weights, bias = children

        return cls.from_parameters(
            in_channels=in_size,
            out_channels=out_size,
            kernel_size=kernel_size,
            weights=weights,
            bias=bias,
            stride=stride,
            padding=padding,
        )
