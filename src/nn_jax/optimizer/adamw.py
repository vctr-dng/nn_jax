import jax.numpy as jnp
from jax import Array, tree_util

from .stateful_optimizer import StatefulOptimizer


@tree_util.register_pytree_node_class
class AdamWState[ParametersT]:
    """First and second moments for a parameter pytree."""

    def __init__(
        self,
        step: Array,
        first_moment: ParametersT,
        second_moment: ParametersT,
    ):
        self.step = step
        self.first_moment = first_moment
        self.second_moment = second_moment

    def tree_flatten(self):
        return (self.step, self.first_moment, self.second_moment), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        step, first_moment, second_moment = children
        return cls(step, first_moment, second_moment)


class AdamW[ParametersT](StatefulOptimizer[ParametersT, AdamWState[ParametersT]]):
    """AdamW with explicit pytree optimizer state."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def init(self, parameters: ParametersT) -> AdamWState[ParametersT]:
        zeros = tree_util.tree_map(jnp.zeros_like, parameters)
        return AdamWState(jnp.array(0, dtype=jnp.int32), zeros, zeros)

    def update(
        self,
        parameters: ParametersT,
        gradients: ParametersT,
        state: AdamWState[ParametersT],
    ) -> tuple[ParametersT, AdamWState[ParametersT]]:
        next_step = state.step + jnp.array(1, dtype=state.step.dtype)
        first_moment = tree_util.tree_map(
            lambda moment, gradient: self.beta1 * moment + (1 - self.beta1) * gradient,
            state.first_moment,
            gradients,
        )
        second_moment = tree_util.tree_map(
            lambda moment, gradient: (
                self.beta2 * moment + (1 - self.beta2) * gradient**2
            ),
            state.second_moment,
            gradients,
        )
        first_bias_correction = 1 - self.beta1**next_step
        second_bias_correction = 1 - self.beta2**next_step
        updated_parameters = tree_util.tree_map(
            lambda parameter, first, second: (
                parameter
                - self.learning_rate
                * (
                    first
                    / first_bias_correction
                    / (jnp.sqrt(second / second_bias_correction) + self.epsilon)
                    + self.weight_decay * parameter
                )
            ),
            parameters,
            first_moment,
            second_moment,
        )
        return updated_parameters, AdamWState(next_step, first_moment, second_moment)
