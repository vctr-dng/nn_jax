import jax
import jax.numpy as jnp
import pytest

from nn_jax.pooling.average_pool2d import AvgPool2D


class TestAvgPool2D:
    in_size = (4, 4, 1)
    pool_size = (2, 2)

    def test_initialization(self):
        pool = AvgPool2D(self.in_size, self.pool_size, stride=2)

        assert pool.in_size == self.in_size
        assert pool.out_size == (2, 2, 1)
        assert pool.pool_size == self.pool_size
        assert pool.stride == 2

    def test_forward_computation(self):
        pool = AvgPool2D(self.in_size, self.pool_size, stride=2)
        inputs = jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4, 1)

        expected_output = jnp.array([[[3.5], [5.5]], [[11.5], [13.5]]])

        assert pool.forward(inputs) == pytest.approx(expected_output)

    def test_pytree_round_trip_preserves_metadata(self):
        pool = AvgPool2D(self.in_size, self.pool_size, stride=2)

        round_tripped = jax.tree.map(lambda value: value, pool)

        assert round_tripped.in_size == pool.in_size
        assert round_tripped.out_size == pool.out_size
        assert round_tripped.pool_size == pool.pool_size
        assert round_tripped.stride == pool.stride

    def test_grad_matches_expected_input_gradient(self):
        pool = AvgPool2D(self.in_size, self.pool_size, stride=2)
        inputs = jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4, 1)

        input_grad = jax.grad(lambda x: jnp.sum(pool.forward(x)))(inputs)

        assert input_grad == pytest.approx(jnp.full(self.in_size, 0.25))

    def test_jit_and_vmap_forward_match_eager(self):
        pool = AvgPool2D(self.in_size, self.pool_size, stride=2)
        inputs = jnp.arange(32, dtype=jnp.float32).reshape(2, *self.in_size)

        compiled_forward = jax.jit(jax.vmap(pool.forward))

        assert compiled_forward(inputs) == pytest.approx(jax.vmap(pool.forward)(inputs))
