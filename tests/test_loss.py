import jax
import jax.numpy as jnp
import pytest

from nn_jax.loss import CategoricalCrossEntropy, MeanSquareError


class TestMeanSquareError:
    def test_derivative_matches_autodiff(self):
        mse = MeanSquareError()
        y_pred = jnp.array([[0.2], [0.7], [0.1]])
        y_true = jnp.array([[0.0], [1.0], [0.0]])

        grad = mse.derivative(y_pred, y_true)

        expected = jax.grad(lambda p: mse.calculate_loss(p, y_true))(y_pred)
        assert grad == pytest.approx(expected)


class TestCategoricalCrossEntropy:
    def test_derivative_matches_autodiff(self):
        cce = CategoricalCrossEntropy()
        y_pred = jnp.array([[0.2], [0.7], [0.1]])
        y_true = jnp.array([[0.0], [1.0], [0.0]])

        grad = cce.derivative(y_pred, y_true)

        expected = jax.grad(lambda p: cce.calculate_loss(p, y_true))(y_pred)
        assert grad == pytest.approx(expected)
