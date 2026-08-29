import jax
import jax.numpy as jnp
import pytest

from nn_jax.activation import ReLU, Softmax, Tanh


class TestReLU:
    def test_backward_matches_autodiff(self):
        relu = ReLU()
        inputs = jnp.array([[-1.0], [0.5], [2.0]])
        out_grad = jnp.array([[1.0], [1.0], [1.0]])

        relu.forward(inputs)
        in_grad = relu.backward(out_grad)

        expected = jax.grad(lambda x: jnp.sum(jnp.maximum(0, x)))(inputs)
        assert in_grad == pytest.approx(expected)


class TestTanh:
    def test_backward_matches_autodiff(self):
        tanh = Tanh()
        inputs = jnp.array([[-1.0], [0.5], [2.0]])
        out_grad = jnp.array([[1.0], [1.0], [1.0]])

        tanh.forward(inputs)
        in_grad = tanh.backward(out_grad)

        expected = jax.grad(lambda x: jnp.sum(jnp.tanh(x)))(inputs)
        assert in_grad == pytest.approx(expected)


class TestSoftmax:
    def test_forward_sums_to_one(self):
        softmax = Softmax()
        inputs = jnp.array([[1.0], [2.0], [3.0]])

        outputs = softmax.forward(inputs)

        assert jnp.sum(outputs) == pytest.approx(1.0)

    def test_backward_matches_autodiff(self):
        softmax = Softmax()
        inputs = jnp.array([[1.0], [-2.0], [0.5]])
        out_grad = jnp.array([[0.3], [-0.7], [1.2]])

        softmax.forward(inputs)
        in_grad = softmax.backward(out_grad)

        def softmax_fn(x):
            exps = jnp.exp(x - jnp.max(x, axis=0, keepdims=True))
            return exps / jnp.sum(exps, axis=0, keepdims=True)

        _, vjp_fn = jax.vjp(softmax_fn, inputs)
        (expected,) = vjp_fn(out_grad)

        assert in_grad == pytest.approx(expected, abs=1e-6)
