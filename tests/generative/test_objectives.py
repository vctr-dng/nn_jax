import jax
import jax.numpy as jnp
import pytest

from nn_jax.generative import conditional_flow_matching_loss


class TestConditionalFlowMatchingLoss:
    def test_oracle_velocity_has_zero_loss(self):
        noise = jnp.array([2.0, -1.0])
        data = jnp.array([6.0, 3.0])
        sigma_min = 0.1
        velocity_target = data - (1 - sigma_min) * noise

        loss = conditional_flow_matching_loss(
            lambda sample, time: velocity_target,
            noise,
            data,
            jnp.array(0.25),
            sigma_min,
        )

        assert loss == pytest.approx(0.0)

    def test_zero_velocity_matches_target_mean_squared_magnitude(self):
        noise = jnp.array([2.0, -1.0])
        data = jnp.array([6.0, 3.0])

        loss = conditional_flow_matching_loss(
            lambda sample, time: jnp.zeros_like(sample),
            noise,
            data,
            jnp.array(0.25),
        )

        assert loss == pytest.approx(jnp.mean(jnp.array([4.0, 4.0]) ** 2))

    def test_is_jittable(self):
        @jax.jit
        def loss(noise, data, time):
            return conditional_flow_matching_loss(
                lambda sample, current_time: data - noise,
                noise,
                data,
                time,
            )

        assert loss(
            jnp.array([0.0]), jnp.array([1.0]), jnp.array(0.5)
        ) == pytest.approx(0.0)
