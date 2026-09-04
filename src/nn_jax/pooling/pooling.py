from jax import Array, tree_util

from ..layer import Layer
from ..utils.im2col import output_spatial_size


@tree_util.register_pytree_node_class
class Pooling(Layer):
    has_params = False

    def __init__(
        self,
        in_size: tuple[int, int, int],
        pool_size: tuple[int, int],
        stride: int = 1,
    ):
        if len(in_size) != 3:
            raise ValueError("in_size must contain height, width, and channels")
        if len(pool_size) != 2:
            raise ValueError("pool_size must contain height and width")
        if any(size <= 0 for size in pool_size):
            raise ValueError("pool_size values must be positive")

        out_size = (
            output_spatial_size(in_size[0], pool_size[0], stride, 0),
            output_spatial_size(in_size[1], pool_size[1], stride, 0),
            in_size[2],
        )

        super().__init__(in_size, out_size)
        self.pool_size = pool_size
        self.stride: int = stride

    def _init_weights(self, key: Array):
        raise NotImplementedError("Pooling layers do not have weights")

    @property
    def _weights_shape(self) -> tuple[int, ...]:
        return ()

    def tree_flatten(
        self,
    ) -> tuple[tuple, tuple[tuple[int, int, int], tuple[int, int], int]]:
        aux_data = (self.in_size, self.pool_size, self.stride)

        return ((), aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[tuple[int, int, int], tuple[int, int], int], _: tuple
    ) -> "Pooling":
        in_size, pool_size, stride = aux_data
        return cls(
            in_size=in_size,
            pool_size=pool_size,
            stride=stride,
        )
