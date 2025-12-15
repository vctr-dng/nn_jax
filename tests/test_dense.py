import jax.numpy as jnp
import pytest

from nn_jax.layer.dense import Dense, single_forward


class TestSingleForward:
    def test_single_sample_forward(self):
        weights = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        biases = jnp.array([0.5, 1.0])  # (2,)
        x = jnp.array([1.0, 2.0])  # (2,)

        result = single_forward(weights, biases, x)

        for i in range(weights.shape[0]):
            calculated_value = weights[i, :] @ x + biases[i]
            assert result[i] == pytest.approx(calculated_value)

        assert result.shape == (weights.shape[0],)

    def test_output_dtype(self):
        weights = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
        biases = jnp.array([0.0], dtype=jnp.float32)
        x = jnp.array([1.0, 1.0], dtype=jnp.float32)

        result = single_forward(weights, biases, x)
        assert result.dtype == jnp.float32


class TestDense:
    def test_initialization(self):
        dense = Dense(input_size=10, output_size=5)

        assert dense.weights.shape == (5, 10)
        assert dense.biases.shape == (5,)
        assert dense.output.shape == (1, 5)
        assert dense.weights.dtype == dense.weight_type
        assert dense.biases.dtype == dense.weight_type

    def test_forward_single_batch(self):
        dense = Dense(input_size=4, output_size=3)
        inputs = jnp.ones((1, 4))

        output = dense.forward(inputs)

        assert output.shape == (1, 3)
        assert isinstance(output, jnp.ndarray)

    def test_forward_multiple_batch(self):
        dense = Dense(input_size=4, output_size=3)
        batch_size = 8
        inputs = jnp.ones((batch_size, 4))

        output = dense.forward(inputs)

        assert output.shape == (batch_size, 3)

    def test_forward_variable_batch_sizes(self):
        dense = Dense(input_size=5, output_size=2)

        # Test different batch sizes work correctly
        for batch_size in [1, 4, 16, 32]:
            inputs = jnp.ones((batch_size, 5))
            output = dense.forward(inputs)
            assert output.shape == (batch_size, 2)

    def test_forward_computation(self):
        dense = Dense(input_size=2, output_size=2)
        # Set known weights and biases
        dense.weights = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        dense.biases = jnp.array([0.5, 1.0])

        inputs = jnp.array([[1.0, 2.0], [2.0, 3.0]])  # 2 samples
        output = dense.forward(inputs)

        for i in range(inputs.shape[0]):
            for j in range(output.shape[1]):
                calculated_value = dense.weights[j, :] @ inputs[i] + dense.biases[j]
                assert output[i, j] == pytest.approx(calculated_value)
