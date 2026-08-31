import jax
import jax.numpy as jnp
import pytest

from nn_jax.activation import ReLU, Softmax, Tanh


class TestReLU:
    def test_vjp_matches_analytical_gradient(self):
        relu = ReLU()
        inputs = jnp.array([-1.0, 0.5, 2.0])
        out_grad = jnp.array([1.0, 1.0, 1.0])

        _, vjp = jax.vjp(relu.forward, inputs)
        (in_grad,) = vjp(out_grad)

        expected = out_grad * jnp.where(inputs > 0, 1, 0)
        assert in_grad == pytest.approx(expected)


class TestTanh:
    def test_vjp_matches_analytical_gradient(self):
        tanh = Tanh()
        inputs = jnp.array([-1.0, 0.5, 2.0])
        out_grad = jnp.array([1.0, 1.0, 1.0])

        _, vjp = jax.vjp(tanh.forward, inputs)
        (in_grad,) = vjp(out_grad)

        expected = out_grad * (1 - jnp.tanh(inputs) ** 2)
        assert in_grad == pytest.approx(expected)


class TestSoftmax:
    def test_forward_sums_to_one_per_batch_element(self):
        softmax = Softmax()
        inputs = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])

        outputs = softmax.forward(inputs)

        assert outputs.shape == inputs.shape
        assert jnp.sum(outputs, axis=-1) == pytest.approx(jnp.ones(inputs.shape[0]))

    def test_vjp_matches_analytical_gradient(self):
        softmax = Softmax()
        inputs = jnp.array([1.0, -2.0, 0.5])
        out_grad = jnp.array([0.3, -0.7, 1.2])

        outputs, vjp = jax.vjp(softmax.forward, inputs)
        (in_grad,) = vjp(out_grad)

        expected = outputs * (out_grad - jnp.sum(out_grad * outputs))

        assert in_grad == pytest.approx(expected, abs=1e-6)
