from jax import Array, tree_util

from ..layer import Layer


@tree_util.register_pytree_node_class
class GlobalAvgPool2D(Layer):
    has_params = False

    def __init__(self, in_size: tuple[int, int, int]):
        if len(in_size) != 3:
            raise ValueError("in_size must contain height, width, and channels")

        super().__init__(in_size, (in_size[-1],))

    def _init_weights(self, key: Array):
        raise NotImplementedError("Global average pooling does not have weights")

    @property
    def _weights_shape(self) -> tuple[()]:
        return ()

    def forward(self, x: Array) -> Array:
        return x.mean(axis=(0, 1))

    def tree_flatten(
        self,
    ) -> tuple[tuple, tuple[tuple[int, int, int]]]:
        aux_data = (self.in_size,)

        return ((), aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[tuple[int, int, int]], _: tuple
    ) -> "GlobalAvgPool2D":
        in_size = aux_data[0]
        return cls(
            in_size=in_size,
        )
