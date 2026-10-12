import jax
import jax.numpy as jnp
import pytest

from nn_jax.generative import TimeConditionedMLP


class TestTimeConditionedMLP:
    def test_predict_returns_configured_output_shape(self):
        model = TimeConditionedMLP(
            input_size=2,
            hidden_sizes=(4, 4),
            output_size=3,
            time_embedding_size=4,
            key=jax.random.key(0),
        )

        prediction = model.predict(jnp.array([1.0, -1.0]), jnp.array(0.5))

        assert prediction.shape == (3,)

    def test_pytree_round_trip_preserves_predictions_and_configuration(self):
        model = TimeConditionedMLP(
            input_size=2,
            hidden_sizes=(4,),
            output_size=2,
            time_embedding_size=4,
            key=jax.random.key(0),
        )
        inputs = jnp.array([1.0, -1.0])
        time = jnp.array(0.5)

        round_tripped = jax.tree.map(lambda value: value, model)

        assert round_tripped.input_size == model.input_size
        assert round_tripped.hidden_sizes == model.hidden_sizes
        assert round_tripped.output_size == model.output_size
        assert round_tripped.time_embedding_size == model.time_embedding_size
        assert round_tripped.predict(inputs, time) == pytest.approx(
            model.predict(inputs, time)
        )

    def test_has_parameter_gradients(self):
        model = TimeConditionedMLP(
            input_size=2,
            hidden_sizes=(4,),
            output_size=2,
            time_embedding_size=4,
            key=jax.random.key(0),
        )

        gradients = jax.grad(
            lambda current_model: jnp.sum(
                current_model.predict(jnp.array([1.0, -1.0]), jnp.array(0.5))
            )
        )(model)

        assert gradients.network.modules[0].weights.shape == (4, 6)
        assert jnp.any(gradients.network.modules[0].weights != 0)

    def test_supports_jitted_external_vmap_batching(self):
        model = TimeConditionedMLP(
            input_size=2,
            hidden_sizes=(4,),
            output_size=2,
            time_embedding_size=4,
            key=jax.random.key(0),
        )
        inputs = jnp.array([[1.0, -1.0], [-0.5, 0.5]])
        times = jnp.array([0.0, 1.0])
        predict_batch = jax.jit(jax.vmap(model.predict))

        predictions = predict_batch(inputs, times)
        expected = jnp.stack(
            [
                model.predict(input_vector, time)
                for input_vector, time in zip(inputs, times)
            ]
        )

        assert predictions == pytest.approx(expected)

    @pytest.mark.parametrize("time_embedding_size", [0, 1, 3])
    def test_requires_a_positive_even_time_embedding_size(self, time_embedding_size):
        with pytest.raises(ValueError, match="time_embedding_size"):
            TimeConditionedMLP(
                input_size=2,
                hidden_sizes=(4,),
                output_size=2,
                time_embedding_size=time_embedding_size,
                key=jax.random.key(0),
            )
