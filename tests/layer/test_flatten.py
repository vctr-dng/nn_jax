import jax
import jax.numpy as jnp
import pytest

from nn_jax.layer.flatten import Flatten


class TestFlatten:
    in_size = (2, 3, 2)

    def test_initialization(self):
        flatten = Flatten(self.in_size)

        assert flatten.in_size == self.in_size
        assert flatten.out_size == 12

    def test_forward_computation(self):
        flatten = Flatten(self.in_size)
        inputs = jnp.arange(12, dtype=jnp.float32).reshape(self.in_size)

        assert flatten.forward(inputs) == pytest.approx(
            jnp.arange(12, dtype=jnp.float32)
        )

    def test_pytree_round_trip_preserves_metadata(self):
        flatten = Flatten(self.in_size)

        round_tripped = jax.tree.map(lambda value: value, flatten)

        assert round_tripped.in_size == flatten.in_size
        assert round_tripped.out_size == flatten.out_size

    def test_grad_matches_expected_input_gradient(self):
        flatten = Flatten(self.in_size)
        inputs = jnp.arange(12, dtype=jnp.float32).reshape(self.in_size)

        input_grad = jax.grad(lambda x: jnp.sum(flatten.forward(x)))(inputs)

        assert input_grad == pytest.approx(jnp.ones(self.in_size))

    def test_jit_and_vmap_forward_match_eager(self):
        flatten = Flatten(self.in_size)
        inputs = jnp.arange(24, dtype=jnp.float32).reshape(2, *self.in_size)

        compiled_forward = jax.jit(jax.vmap(flatten.forward))

        assert compiled_forward(inputs) == pytest.approx(
            jax.vmap(flatten.forward)(inputs)
        )
