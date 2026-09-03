from math import prod

from jax import Array, tree_util

from .layer import Layer


@tree_util.register_pytree_node_class
class Flatten(Layer):
    has_params = False

    def __init__(
        self,
        in_size: tuple[int, ...],
    ):
        out_size: int = prod(in_size)
        super().__init__(in_size, out_size)

    def _init_weights(self, key: Array):
        raise NotImplementedError("Flatten layer does not have weights")

    @property
    def _weights_shape(self) -> tuple[int, ...]:
        return ()

    def forward(self, x: Array) -> Array:
        return x.reshape(-1)

    def tree_flatten(self) -> tuple[tuple, tuple[tuple[int, ...]]]:
        aux_data = (self.in_size,)

        return ((), aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[tuple[int, ...]], _: tuple) -> "Flatten":
        in_size = aux_data[0]

        return cls(in_size=in_size)
