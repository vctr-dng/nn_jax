import jax
import jax.numpy as jnp
import pytest

from nn_jax.optimizer import AdamW
from nn_jax.training import TrainState


class TestTrainState:
    def test_is_a_jittable_pytree(self):
        parameters = {"weights": jnp.array([1.0])}
        optimizer_state = AdamW().init(parameters)
        state = TrainState(
            parameters=parameters,
            optimizer_state=optimizer_state,
            ema_parameters=parameters,
            rng_key=jax.random.key(0),
            step=jnp.array(0, dtype=jnp.int32),
        )

        @jax.jit
        def advance(current_state):
            return TrainState(
                parameters=current_state.parameters,
                optimizer_state=current_state.optimizer_state,
                ema_parameters=current_state.ema_parameters,
                rng_key=current_state.rng_key,
                step=current_state.step + 1,
            )

        advanced_state = advance(state)

        assert advanced_state.step == 1
        assert advanced_state.parameters["weights"] == pytest.approx(jnp.array([1.0]))
