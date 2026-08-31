import jax
import jax.numpy as jnp
import pytest

from nn_jax.activation import Softmax, Tanh
from nn_jax.layer import Dense
from nn_jax.loss import MeanSquareError
from nn_jax.sequential import Sequential


class TestSequential:
    def test_grad_matches_explicit_loss(self):
        d1 = Dense(3, 4, jax.random.key(0))
        d2 = Dense(4, 2, jax.random.key(1))
        network = Sequential([d1, Tanh(), d2, Softmax()])

        x = jnp.array([0.5, -0.2, 1.0])
        y_true = jnp.array([1.0, 0.0])
        mse = MeanSquareError()

        def model_loss(model):
            return mse(model.forward(x), y_true)

        grads = jax.grad(model_loss)(network)

        def pure_forward(w1, b1, w2, b2, x):
            h = jnp.tanh(w1 @ x + b1)
            logits = w2 @ h + b2
            exps = jnp.exp(logits - jnp.max(logits))
            return exps / jnp.sum(exps)

        def reference_loss(w1, b1, w2, b2):
            pred = pure_forward(w1, b1, w2, b2, x)
            return jnp.mean((pred - y_true) ** 2)

        expected_grads = jax.grad(reference_loss, argnums=(0, 1, 2, 3))(
            d1.weights, d1.bias, d2.weights, d2.bias
        )

        assert grads.modules[0].weights == pytest.approx(expected_grads[0], abs=1e-5)
        assert grads.modules[0].bias == pytest.approx(expected_grads[1], abs=1e-5)
        assert grads.modules[2].weights == pytest.approx(expected_grads[2], abs=1e-5)
        assert grads.modules[2].bias == pytest.approx(expected_grads[3], abs=1e-5)

    def test_jit_and_vmap_forward(self):
        network = Sequential(
            [
                Dense(2, 3, jax.random.key(0)),
                Tanh(),
                Dense(3, 1, jax.random.key(1)),
            ]
        )
        inputs = jnp.array([[0.0, 1.0], [1.0, 0.0]])

        def forward_one(model, x):
            return model.forward(x)

        batched_forward = jax.jit(jax.vmap(forward_one, in_axes=(None, 0)))
        expected = jnp.stack([network.forward(x) for x in inputs])

        assert batched_forward(network, inputs) == pytest.approx(expected)
