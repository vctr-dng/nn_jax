import jax.numpy as jnp
import jax.random as random
from jax.typing import ArrayLike

from .layer import Layer

SEED = 0
key = random.key(SEED)


class Dense(Layer):
    weight_type = jnp.float32

    def __init__(self, input_size: ArrayLike, output_size: ArrayLike):
        super().__init__(input_size, output_size)
        self.weights: ArrayLike = 0.1 * random.normal(
            key, shape=(self.output_size, self.input_size), dtype=self.weight_type
        )
        self.biases: ArrayLike = jnp.zeros(
            (1, self.output_size), dtype=self.weight_type
        )
        self.output: ArrayLike = self.forward(jnp.zeros((1, self.input_size)))

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        output = jnp.dot(inputs, self.weights.T) + self.biases
        return output


# if __name__ == "__main__":
#     # Create a dense layer with 3 input features and 2 output features
#     n_inputs = 3
#     n_neurons = 2
#     layer1 = Dense(n_inputs, n_neurons)
#     layer1.weights = jnp.ones((n_inputs, n_neurons))
#     print(layer1.weights)
#     x = jnp.array([[1, 1, 1], [2, 2, 2]], dtype=jnp.float32)
#     print(layer1.forward(x))
