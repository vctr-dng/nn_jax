import jax
import jax.numpy as jnp
import pytest

from nn_jax.generative import integrate_ode, sample_euler, sample_midpoint


class TestOdeSamplers:
    def test_euler_integrates_a_constant_vector_field_and_records_trajectory(self):
        initial_samples = jnp.zeros((2, 1))

        def constant_velocity(samples, time):
            del time
            return jnp.ones_like(samples)

        final_samples, trajectory = sample_euler(
            constant_velocity, initial_samples, num_steps=4
        )

        assert final_samples == pytest.approx(jnp.ones((2, 1)))
        assert trajectory.shape == (5, 2, 1)
        assert trajectory[:, 0, 0] == pytest.approx(jnp.linspace(0.0, 1.0, 5))

    def test_midpoint_uses_the_midpoint_time_for_time_dependent_velocity(self):
        initial_samples = jnp.zeros((3, 2))

        def time_velocity(samples, time):
            return jnp.full_like(samples, time)

        midpoint_samples, _ = sample_midpoint(
            time_velocity, initial_samples, num_steps=4
        )
        euler_samples, _ = sample_euler(time_velocity, initial_samples, num_steps=4)

        assert midpoint_samples == pytest.approx(jnp.full((3, 2), 0.5))
        assert euler_samples == pytest.approx(jnp.full((3, 2), 0.375))

    def test_midpoint_is_more_accurate_than_euler_for_linear_velocity(self):
        initial_samples = jnp.ones((1, 1))

        def linear_velocity(samples, time):
            del time
            return samples

        euler_samples, _ = sample_euler(linear_velocity, initial_samples, num_steps=8)
        midpoint_samples, _ = sample_midpoint(
            linear_velocity, initial_samples, num_steps=8
        )
        exact = jnp.full((1, 1), jnp.e)

        assert (
            jnp.abs(midpoint_samples - exact).max()
            < jnp.abs(euler_samples - exact).max()
        )

    def test_integration_is_jittable(self):
        initial_samples = jnp.zeros((2, 1))

        def constant_velocity(samples, time):
            del time
            return jnp.ones_like(samples)

        compiled = jax.jit(
            lambda samples: integrate_ode(
                constant_velocity, samples, num_steps=4, method="midpoint"
            )
        )

        compiled_samples, compiled_trajectory = compiled(initial_samples)
        expected_samples, expected_trajectory = integrate_ode(
            constant_velocity, initial_samples, num_steps=4, method="midpoint"
        )

        assert compiled_samples == pytest.approx(expected_samples)
        assert compiled_trajectory == pytest.approx(expected_trajectory)

    @pytest.mark.parametrize("num_steps", [0, -1])
    def test_requires_a_positive_step_count(self, num_steps):
        with pytest.raises(ValueError, match="num_steps must be positive"):
            sample_euler(
                lambda samples, time: samples + time, jnp.zeros((1, 1)), num_steps
            )

    def test_requires_a_supported_method(self):
        with pytest.raises(ValueError, match="method must be 'euler' or 'midpoint'"):
            integrate_ode(
                lambda samples, time: samples + time,
                jnp.zeros((1, 1)),
                num_steps=1,
                method="heun",  # type: ignore[arg-type]
            )
