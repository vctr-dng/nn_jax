import jax
import jax.numpy as jnp
import pytest

from nn_jax.activation import Softmax, Tanh
from nn_jax.layer import Dense
from nn_jax.loss import MeanSquareError
from nn_jax.sequential import Sequential


class TestSequential:
    def test_backward_matches_autodiff(self):
        d1 = Dense(3, 4)
        d2 = Dense(4, 2)
        network = Sequential([d1, Tanh(), d2, Softmax()])

        x = jnp.array([[0.5], [-0.2], [1.0]])
        y_true = jnp.array([[1.0], [0.0]])
        mse = MeanSquareError()

        y_pred = network.forward(x)
        network.backward(mse.derivative(y_pred, y_true))

        def pure_forward(w1, b1, w2, b2, x):
            h = jnp.tanh(jnp.dot(w1, x) + b1)
            logits = jnp.dot(w2, h) + b2
            exps = jnp.exp(logits - jnp.max(logits, axis=0, keepdims=True))
            return exps / jnp.sum(exps, axis=0, keepdims=True)

        def loss_fn(w1, b1, w2, b2):
            pred = pure_forward(w1, b1, w2, b2, x)
            return jnp.mean((pred - y_true) ** 2)

        expected_grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3))(
            d1.weights, d1.bias, d2.weights, d2.bias
        )

        assert d1.w_grad == pytest.approx(expected_grads[0], abs=1e-5)
        assert d1.b_grad == pytest.approx(expected_grads[1], abs=1e-5)
        assert d2.w_grad == pytest.approx(expected_grads[2], abs=1e-5)
        assert d2.b_grad == pytest.approx(expected_grads[3], abs=1e-5)
