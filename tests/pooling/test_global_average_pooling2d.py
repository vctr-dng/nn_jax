import jax
import jax.numpy as jnp
import pytest

from nn_jax.pooling.global_average_pooling2d import GlobalAvgPool2D


class TestGlobalAvgPool2D:
    in_size = (2, 3, 2)

    def test_initialization(self):
        pool = GlobalAvgPool2D(self.in_size)

        assert pool.in_size == self.in_size
        assert pool.out_size == (self.in_size[-1],)

    def test_forward_computation(self):
        pool = GlobalAvgPool2D(self.in_size)
        inputs = jnp.arange(12, dtype=jnp.float32).reshape(self.in_size)

        assert pool.forward(inputs) == pytest.approx(jnp.array([5.0, 6.0]))

    def test_pytree_round_trip_preserves_metadata(self):
        pool = GlobalAvgPool2D(self.in_size)

        round_tripped = jax.tree.map(lambda value: value, pool)

        assert round_tripped.in_size == pool.in_size
        assert round_tripped.out_size == pool.out_size

    def test_grad_matches_expected_input_gradient(self):
        pool = GlobalAvgPool2D(self.in_size)
        inputs = jnp.arange(12, dtype=jnp.float32).reshape(self.in_size)

        input_grad = jax.grad(lambda x: jnp.sum(pool.forward(x)))(inputs)

        assert input_grad == pytest.approx(jnp.full(self.in_size, 1 / 6))

    def test_jit_and_vmap_forward_match_eager(self):
        pool = GlobalAvgPool2D(self.in_size)
        inputs = jnp.arange(24, dtype=jnp.float32).reshape(2, *self.in_size)

        compiled_forward = jax.jit(jax.vmap(pool.forward))

        assert compiled_forward(inputs) == pytest.approx(jax.vmap(pool.forward)(inputs))
