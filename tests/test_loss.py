import jax
import jax.numpy as jnp
import pytest

from nn_jax.loss import CategoricalCrossEntropy, MeanSquareError


class TestMeanSquareError:
    def test_is_differentiable(self):
        mse = MeanSquareError()
        y_pred = jnp.array([[0.2], [0.7], [0.1]])
        y_true = jnp.array([[0.0], [1.0], [0.0]])

        grad = jax.grad(lambda prediction: mse(prediction, y_true))(y_pred)

        expected = 2 * (y_pred - y_true) / y_true.size
        assert grad == pytest.approx(expected)


class TestCategoricalCrossEntropy:
    def test_returns_mean_negative_log_likelihood_per_example(self):
        cce = CategoricalCrossEntropy()
        pred = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        target = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        expected = -jnp.mean(jnp.array([jnp.log(0.7), jnp.log(0.6)]))

        assert cce(pred, target) == pytest.approx(expected)

    def test_is_invariant_to_duplicate_batch_examples(self):
        cce = CategoricalCrossEntropy()
        pred = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        target = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        duplicated_pred = jnp.concatenate([pred, pred])
        duplicated_target = jnp.concatenate([target, target])

        assert cce(duplicated_pred, duplicated_target) == pytest.approx(
            cce(pred, target)
        )

    def test_gradient_is_meaned_over_the_batch(self):
        cce = CategoricalCrossEntropy()
        pred = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        target = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        grad = jax.grad(lambda prediction: cce(prediction, target))(pred)
        expected = -target / (pred.shape[0] * pred)

        assert grad == pytest.approx(expected)
