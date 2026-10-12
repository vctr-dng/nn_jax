import jax
import jax.numpy as jnp
import pytest

from nn_jax.training import initialize_ema, update_ema


class TestEma:
    def test_initializes_an_equal_parameter_pytree(self):
        parameters = {
            "weights": jnp.array([1.0, -2.0]),
            "bias": jnp.array([0.5]),
        }

        ema_parameters = initialize_ema(parameters)

        assert ema_parameters["weights"] == pytest.approx(parameters["weights"])
        assert ema_parameters["bias"] == pytest.approx(parameters["bias"])

    def test_updates_each_parameter_by_decay_interpolation(self):
        ema_parameters = {
            "weights": jnp.array([0.0, 10.0]),
            "bias": jnp.array([2.0]),
        }
        parameters = {
            "weights": jnp.array([4.0, -2.0]),
            "bias": jnp.array([6.0]),
        }

        updated_ema_parameters = update_ema(ema_parameters, parameters, decay=0.75)

        assert updated_ema_parameters["weights"] == pytest.approx(jnp.array([1.0, 7.0]))
        assert updated_ema_parameters["bias"] == pytest.approx(jnp.array([3.0]))
        assert ema_parameters["weights"] == pytest.approx(jnp.array([0.0, 10.0]))

    def test_preserves_parameter_dtype(self):
        parameters = {"weights": jnp.array([1.0], dtype=jnp.bfloat16)}

        initialized_ema_parameters = initialize_ema(parameters)
        updated_ema_parameters = update_ema(
            initialized_ema_parameters,
            {"weights": jnp.array([2.0], dtype=jnp.bfloat16)},
            decay=0.5,
        )

        assert initialized_ema_parameters["weights"].dtype == jnp.bfloat16
        assert updated_ema_parameters["weights"].dtype == jnp.bfloat16

    def test_is_jittable(self):
        update = jax.jit(lambda ema, parameters: update_ema(ema, parameters, decay=0.9))

        updated_ema_parameters = update(
            {"weights": jnp.array([0.0])},
            {"weights": jnp.array([10.0])},
        )

        assert updated_ema_parameters["weights"] == pytest.approx(jnp.array([1.0]))

    @pytest.mark.parametrize("decay", [-0.1, 1.1])
    def test_rejects_decay_outside_closed_unit_interval(self, decay):
        with pytest.raises(ValueError, match="decay"):
            update_ema(jnp.array([0.0]), jnp.array([1.0]), decay)
