from jax import Array, tree_util

from .layer import Layer


@tree_util.register_pytree_node_class
class Dense(Layer):
    weights: Array
    bias: Array

    # def __init__(
    #     self,
    #     in_size: int,
    #     out_size: int,
    #     key: Array | None = None,
    #     weights: Array | None = None,
    #     bias: Array | None = None,
    # ):
    #     super().__init__(in_size, out_size, key, weights, bias)

    @classmethod
    def from_parameters(
        cls, in_size: int, out_size: int, weights: Array, bias: Array
    ) -> "Dense":
        return cls(in_size, out_size, weights=weights, bias=bias)

    def forward(self, x: Array) -> Array:
        return self.weights @ x + self.bias

    def tree_flatten(self):
        children = (
            self.weights,
            self.bias,
        )
        aux_data = (self.in_size, self.out_size)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        in_size, out_size = aux_data
        weights, bias = children

        return cls.from_parameters(in_size, out_size, weights, bias)
