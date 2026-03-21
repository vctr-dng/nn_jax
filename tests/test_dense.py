import jax.numpy as jnp
import pytest

from nn_jax.layer.dense import Dense


class TestDense:
    in_size = 10
    out_size = 5

    def test_initialization(self):
        dense = Dense(in_size=self.in_size, out_size=self.out_size)

        assert dense.weights.shape == (self.out_size, self.in_size)
        assert dense.bias.shape == (self.out_size, 1)
        assert dense.outputs.shape == (self.out_size, 1)
        assert dense.weights.dtype == dense.weight_type
        assert dense.bias.dtype == dense.weight_type

    def test_forward(self):
        dense = Dense(in_size=self.in_size, out_size=self.out_size)
        inputs = jnp.ones((self.in_size, 1))

        output = dense.forward(inputs)

        assert output.shape == (self.out_size, 1)
        assert isinstance(output, jnp.ndarray)

    def test_forward_computation(self):
        dense = Dense(in_size=2, out_size=2)
        # Set known weights and biases
        dense.weights = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        dense.bias = jnp.array([0.5, 1.0])

        inputs = jnp.array([[1.0, 2.0], [2.0, 3.0]])  # 2 samples

        for i in range(inputs.shape[0]):
            x = inputs[i]
            expected_output = jnp.dot(dense.weights, x) + dense.bias
            assert dense.forward(x) == pytest.approx(expected_output)
