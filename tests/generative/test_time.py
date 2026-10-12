import jax.numpy as jnp
import pytest

from nn_jax.generative import sinusoidal_time_embedding


class TestSinusoidalTimeEmbedding:
    def test_zero_time_has_zero_sine_and_unit_cosine_components(self):
        embedding = sinusoidal_time_embedding(jnp.array(0.0), embedding_size=4)

        assert embedding == pytest.approx(jnp.array([0.0, 0.0, 1.0, 1.0]))

    def test_uses_geometrically_spaced_frequencies(self):
        embedding = sinusoidal_time_embedding(jnp.array(2.0), embedding_size=4)

        expected = jnp.array([jnp.sin(2.0), jnp.sin(0.02), jnp.cos(2.0), jnp.cos(0.02)])
        assert embedding == pytest.approx(expected)

    def test_preserves_leading_time_dimensions(self):
        embedding = sinusoidal_time_embedding(jnp.array([0.0, 1.0]), embedding_size=4)

        assert embedding.shape == (2, 4)

    @pytest.mark.parametrize("embedding_size", [0, 1, 3])
    def test_requires_a_positive_even_embedding_size(self, embedding_size):
        with pytest.raises(ValueError, match="embedding_size"):
            sinusoidal_time_embedding(jnp.array(0.0), embedding_size)
