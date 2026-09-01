from jax import Array, random, tree_util

from .layer import Layer


@tree_util.register_pytree_node_class
class Dense(Layer):
    weights: Array
    bias: Array

    def _init_weights(self, key: Array):
        self.weights = 0.1 * random.normal(key, (self.out_size, self.in_size))

    @property
    def _weight_shape(self) -> tuple[int, ...]:
        return (self.out_size, self.in_size)

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
