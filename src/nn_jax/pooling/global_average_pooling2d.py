from jax import Array, tree_util

from .pooling import Pooling


@tree_util.register_pytree_node_class
class GlobalAvgPool2D(Pooling):
    def __init__(self, in_size: tuple[int, ...]):
        pool_size = (in_size[0], in_size[1])
        super().__init__(in_size, pool_size, 1)
        self.out_size: tuple[int] = (self.out_size[-1],)

    def forward(self, x: Array) -> Array:
        return x.mean(axis=(0, 1))

    def tree_flatten(
        self,
    ) -> tuple[tuple, tuple[tuple[int, ...], tuple[int, ...], int]]:
        aux_data = (self.in_size,)

        return ((), aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[tuple[int, ...]], _: tuple
    ) -> "GlobalAvgPool2D":
        in_size = aux_data[0]
        return cls(
            in_size=in_size,
        )
