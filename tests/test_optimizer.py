import jax
import jax.numpy as jnp
import pytest

from nn_jax.activation import Tanh
from nn_jax.layer import Dense
from nn_jax.loss import MeanSquareError
from nn_jax.optimizer import StochasticGradientDescent
from nn_jax.sequential import Sequential


class TestStochasticGradientDescent:
    def test_jitted_step_updates_parameters_without_mutating_model(self):
        model = Sequential(
            [
                Dense.from_parameters(
                    1,
                    1,
                    weights=jnp.zeros((1, 1)),
                    bias=jnp.zeros((1,)),
                ),
                Tanh(),
            ]
        )
        optimizer = StochasticGradientDescent(learning_rate=0.1)
        loss = MeanSquareError()
        inputs = jnp.array([1.0])
        targets = jnp.array([1.0])

        def loss_fn(current_model):
            return loss(current_model.forward(inputs), targets)

        @jax.jit
        def train_step(current_model):
            loss_value, grads = jax.value_and_grad(loss_fn)(current_model)
            return optimizer(current_model, grads), loss_value

        updated_model, initial_loss = train_step(model)

        assert isinstance(updated_model.modules[1], Tanh)
        assert updated_model.modules[0].weights == pytest.approx(jnp.array([[0.2]]))
        assert updated_model.modules[0].bias == pytest.approx(jnp.array([0.2]))
        assert model.modules[0].weights == pytest.approx(jnp.zeros((1, 1)))
        assert model.modules[0].bias == pytest.approx(jnp.zeros((1,)))
        assert loss_fn(updated_model) < initial_loss
