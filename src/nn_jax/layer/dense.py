import jax.numpy as jnp
import jax.random as random

from jax import Array, tree_util

from .layer import Layer

SEED = 0
key = random.key(SEED)


class Dense(Layer):
    weight_type = jnp.float32

    def __init__(self, in_size: int, out_size: int):
        super().__init__(in_size, out_size)
        self.weights: Array = 0.1 * random.normal(
            key, shape=(self.out_size, self.in_size), dtype=self.weight_type
        )
        self.bias: Array = jnp.zeros(shape=(self.out_size, 1), dtype=self.weight_type)
        self.w_grad: Array = jnp.zeros_like(self.weights)
        self.b_grad: Array = jnp.zeros_like(self.bias)

    def forward(self, inputs: Array) -> Array:
        super().forward(inputs)
        self.outputs = jnp.dot(self.weights, inputs) + self.bias
        return self.outputs

    def backward(self, out_grad: Array) -> Array:
        self.w_grad = jnp.dot(out_grad, self.inputs.T)
        self.b_grad = out_grad
        in_grad = jnp.dot(self.weights.T, out_grad)

        return in_grad

    @property
    def parameters(self) -> list[Array]:
        return [self.weights, self.bias]

    @parameters.setter
    def parameters(self, new_parameters: list[Array]):
        self.weights = new_parameters[0]
        self.bias = new_parameters[1]

    @property
    def gradients(self) -> list[Array]:
        return [self.w_grad, self.b_grad]

    def zero_grad(self):
        self.w_grad = jnp.zeros_like(self.w_grad)
        self.b_grad = jnp.zeros_like(self.b_grad)

    def _tree_flatten(self):
        dynamic_values = (
            self.weights,
            self.bias,
            self.inputs,
            self.outputs,
            self.w_grad,
            self.b_grad,
        )
        static_values = {
            "in_size": self.in_size,
            "out_size": self.out_size,
        }
        return (dynamic_values, static_values)

    @classmethod
    def _tree_unflatten(cls, static_values, dynamic_values):
        dense = cls(
            static_values["in_size"],
            static_values["out_size"],
        )
        dense.weights = dynamic_values[0]
        dense.bias = dynamic_values[1]
        dense.inputs = dynamic_values[2]
        dense.outputs = dynamic_values[3]
        dense.w_grad = dynamic_values[4]
        dense.b_grad = dynamic_values[5]
        return dense


tree_util.register_pytree_node(
    Dense,
    Dense._tree_flatten,
    Dense._tree_unflatten,
)
