import jax
import jax.numpy as jnp
import pytest

from nn_jax.layer import Dense
from nn_jax.optimizer import AdamW


class TestAdamW:
    def test_initializes_zero_moments_matching_parameter_pytree(self):
        parameters = {"weights": jnp.array([1.0, -2.0])}

        state = AdamW().init(parameters)

        assert state.step == 0
        assert state.first_moment["weights"] == pytest.approx(jnp.zeros(2))
        assert state.second_moment["weights"] == pytest.approx(jnp.zeros(2))

    def test_first_update_matches_adamw_equation(self):
        parameters = {"weights": jnp.array([1.0])}
        gradients = {"weights": jnp.array([0.5])}
        optimizer = AdamW(learning_rate=0.1, weight_decay=0.01)

        updated_parameters, state = optimizer.update(
            parameters, gradients, optimizer.init(parameters)
        )

        assert state.step == 1
        assert state.first_moment["weights"] == pytest.approx(jnp.array([0.05]))
        assert state.second_moment["weights"] == pytest.approx(jnp.array([0.00025]))
        assert updated_parameters["weights"] == pytest.approx(jnp.array([0.899]))
        assert parameters["weights"] == pytest.approx(jnp.array([1.0]))

    def test_jitted_update_reduces_loss_without_mutating_parameters(self):
        parameters = Dense.from_parameters(
            1,
            1,
            weights=jnp.array([[0.0]]),
            bias=jnp.array([0.0]),
        )
        optimizer = AdamW(learning_rate=0.1)
        optimizer_state = optimizer.init(parameters)
        inputs = jnp.array([1.0])
        target = jnp.array([1.0])

        def loss_fn(current_parameters):
            return jnp.mean((current_parameters.forward(inputs) - target) ** 2)

        @jax.jit
        def train_step(current_parameters, current_optimizer_state):
            loss, gradients = jax.value_and_grad(loss_fn)(current_parameters)
            next_parameters, next_optimizer_state = optimizer.update(
                current_parameters, gradients, current_optimizer_state
            )
            return next_parameters, next_optimizer_state, loss

        updated_parameters, updated_optimizer_state, initial_loss = train_step(
            parameters, optimizer_state
        )

        assert updated_optimizer_state.step == 1
        assert parameters.weights == pytest.approx(jnp.array([[0.0]]))
        assert parameters.bias == pytest.approx(jnp.array([0.0]))
        assert loss_fn(updated_parameters) < initial_loss
