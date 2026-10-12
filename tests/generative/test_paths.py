import jax.numpy as jnp
import pytest

from nn_jax.generative import conditional_flow_matching_path


class TestConditionalFlowMatchingPath:
    def test_standard_path_starts_at_noise_and_ends_at_data(self):
        noise = jnp.array([2.0, -1.0])
        data = jnp.array([6.0, 3.0])

        initial_sample, initial_velocity = conditional_flow_matching_path(
            noise, data, jnp.array(0.0)
        )
        final_sample, final_velocity = conditional_flow_matching_path(
            noise, data, jnp.array(1.0)
        )

        expected_velocity = jnp.array([4.0, 4.0])
        assert initial_sample == pytest.approx(noise)
        assert final_sample == pytest.approx(data)
        assert initial_velocity == pytest.approx(expected_velocity)
        assert final_velocity == pytest.approx(expected_velocity)

    def test_finite_noise_endpoint_and_velocity_match_analytical_path(self):
        noise = jnp.array([2.0, -1.0])
        data = jnp.array([6.0, 3.0])

        initial_sample, velocity = conditional_flow_matching_path(
            noise, data, jnp.array(0.0), sigma_min=0.1
        )
        final_sample, _ = conditional_flow_matching_path(
            noise, data, jnp.array(1.0), sigma_min=0.1
        )

        assert initial_sample == pytest.approx(noise)
        assert final_sample == pytest.approx(jnp.array([6.2, 2.9]))
        assert velocity == pytest.approx(jnp.array([4.2, 3.9]))

    @pytest.mark.parametrize("sigma_min", [-0.1, 1.0])
    def test_rejects_invalid_sigma_min(self, sigma_min):
        with pytest.raises(ValueError, match="sigma_min"):
            conditional_flow_matching_path(
                jnp.array([0.0]), jnp.array([1.0]), jnp.array(0.5), sigma_min
            )
