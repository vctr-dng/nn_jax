import jax.numpy as jnp
from jax import Array, random, tree_util

from .layer import Layer


@tree_util.register_pytree_node_class
class Dense(Layer):
    weights: Array
    bias: Array

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: Array | None = None,
        weights: Array | None = None,
        bias: Array | None = None,
    ):
        super().__init__(in_size, out_size)

        if key is not None and weights is not None:
            raise ValueError("Key and weights cannot be provided together")

        if weights is None:
            if key is None:
                raise ValueError("Key is required if weights are not provided")
            self._init_weights(key)
        else:
            if weights.shape != (self.out_size, self.in_size):
                raise ValueError(
                    f"Weights shape must be ({self.out_size}, {self.in_size}), got {weights.shape}"
                )
            self.weights = weights

        if bias is not None:
            if bias.shape != (self.out_size,):
                raise ValueError(
                    f"Bias shape must be ({self.out_size},), got {bias.shape}"
                )
            self.bias = bias
        else:
            self.bias = jnp.zeros((self.out_size,))

    def _init_weights(self, key: Array):
        self.weights = 0.1 * random.normal(key, (self.out_size, self.in_size))

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
