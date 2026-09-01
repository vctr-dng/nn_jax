import jax
import jax.numpy as jnp
import pytest

from nn_jax.layer.dense import Dense


class TestDense:
    in_size = 10
    out_size = 5

    def test_initialization(self):
        dense = Dense(
            in_size=self.in_size,
            out_size=self.out_size,
            key=jax.random.key(0),
        )

        assert dense.weights.shape == (self.out_size, self.in_size)
        assert dense.bias.shape == (self.out_size,)

    def test_initialization_requires_exactly_one_weight_source(self):
        weights = jnp.zeros((self.out_size, self.in_size))

        with pytest.raises(ValueError, match="Key is required"):
            Dense(self.in_size, self.out_size)

        with pytest.raises(ValueError, match="cannot be provided together"):
            Dense(self.in_size, self.out_size, jax.random.key(0), weights)

    def test_initialization_validates_parameter_shapes(self):
        with pytest.raises(ValueError, match="Weights shape"):
            Dense(self.in_size, self.out_size, weights=jnp.zeros((1, 1)))

        with pytest.raises(ValueError, match="Bias shape"):
            Dense(
                self.in_size,
                self.out_size,
                weights=jnp.zeros((self.out_size, self.in_size)),
                bias=jnp.zeros((1,)),
            )

    def test_forward(self):
        dense = Dense(self.in_size, self.out_size, jax.random.key(0))
        inputs = jnp.ones(self.in_size)

        output = dense.forward(inputs)

        assert output.shape == (self.out_size,)
        assert isinstance(output, jnp.ndarray)

    def test_forward_computation(self):
        dense = Dense.from_parameters(
            in_size=2,
            out_size=2,
            weights=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            bias=jnp.array([0.5, 1.0]),
        )

        inputs = jnp.array([1.0, 2.0])
        expected_output = jnp.dot(dense.weights, inputs) + dense.bias

        assert dense.forward(inputs) == pytest.approx(expected_output)

    def test_pytree_round_trip_preserves_parameters_and_metadata(self):
        dense = Dense(self.in_size, self.out_size, jax.random.key(0))

        round_tripped = jax.tree.map(lambda value: value, dense)

        assert round_tripped.in_size == dense.in_size
        assert round_tripped.out_size == dense.out_size
        assert round_tripped.weights == pytest.approx(dense.weights)
        assert round_tripped.bias == pytest.approx(dense.bias)

    def test_grad_matches_analytical_gradient(self):
        dense = Dense(self.in_size, self.out_size, jax.random.key(0))
        inputs = jnp.ones(self.in_size)
        out_grad = jnp.arange(self.out_size, dtype=jnp.float32)

        def weighted_output(layer, x):
            return jnp.vdot(layer.forward(x), out_grad)

        dense_grad, input_grad = jax.grad(weighted_output, argnums=(0, 1))(
            dense, inputs
        )

        expected_weights_grad = jnp.outer(out_grad, inputs)
        expected_bias_grad = out_grad
        expected_input_grad = dense.weights.T @ out_grad

        assert dense_grad.weights == pytest.approx(expected_weights_grad)
        assert dense_grad.bias == pytest.approx(expected_bias_grad)
        assert input_grad == pytest.approx(expected_input_grad)

    def test_jit_forward_matches_eager(self):
        dense = Dense(self.in_size, self.out_size, jax.random.key(0))
        inputs = jnp.ones(self.in_size)

        compiled_forward = jax.jit(lambda layer, x: layer.forward(x))

        assert compiled_forward(dense, inputs) == pytest.approx(dense.forward(inputs))
