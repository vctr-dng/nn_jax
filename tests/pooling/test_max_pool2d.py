import jax
import jax.numpy as jnp
import pytest

from nn_jax.pooling.max_pool2d import MaxPool2D


class TestMaxPool2D:
    in_size = (4, 4, 1)
    pool_size = (2, 2)

    def test_initialization(self):
        pool = MaxPool2D(self.in_size, self.pool_size, stride=2)

        assert pool.in_size == self.in_size
        assert pool.out_size == (2, 2, 1)
        assert pool.pool_size == self.pool_size
        assert pool.stride == 2

    @pytest.mark.parametrize(
        ("in_size", "pool_size"),
        [
            ((4, 4), (2, 2)),
            ((4, 4, 1), (2,)),
            ((4, 4, 1), (2, 2, 2)),
            ((4, 4, 1), (0, 2)),
        ],
    )
    def test_initialization_rejects_invalid_geometry(self, in_size, pool_size):
        with pytest.raises(ValueError):
            MaxPool2D(in_size, pool_size)

    def test_forward_computation(self):
        pool = MaxPool2D(self.in_size, self.pool_size, stride=2)
        inputs = jnp.array(
            [
                [[1.0], [3.0], [2.0], [4.0]],
                [[5.0], [7.0], [6.0], [8.0]],
                [[9.0], [11.0], [10.0], [12.0]],
                [[13.0], [15.0], [14.0], [16.0]],
            ]
        )

        expected_output = jnp.array([[[7.0], [8.0]], [[15.0], [16.0]]])

        assert pool.forward(inputs) == pytest.approx(expected_output)

    def test_pytree_round_trip_preserves_metadata(self):
        pool = MaxPool2D(self.in_size, self.pool_size, stride=2)

        round_tripped = jax.tree.map(lambda value: value, pool)

        assert round_tripped.in_size == pool.in_size
        assert round_tripped.out_size == pool.out_size
        assert round_tripped.pool_size == pool.pool_size
        assert round_tripped.stride == pool.stride

    def test_grad_matches_expected_input_gradient(self):
        pool = MaxPool2D(self.in_size, self.pool_size, stride=2)
        inputs = jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4, 1)

        input_grad = jax.grad(lambda x: jnp.sum(pool.forward(x)))(inputs)

        expected_input_grad = jnp.array(
            [
                [[0.0], [0.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0], [1.0]],
                [[0.0], [0.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0], [1.0]],
            ]
        )
        assert input_grad == pytest.approx(expected_input_grad)

    def test_jit_and_vmap_forward_match_eager(self):
        pool = MaxPool2D(self.in_size, self.pool_size, stride=2)
        inputs = jnp.arange(32, dtype=jnp.float32).reshape(2, *self.in_size)

        compiled_forward = jax.jit(jax.vmap(pool.forward))

        assert compiled_forward(inputs) == pytest.approx(jax.vmap(pool.forward)(inputs))
