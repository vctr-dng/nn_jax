from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

VelocityFunction = Callable[[Array, Array], Array]


def integrate_ode(
    velocity_function: VelocityFunction,
    initial_samples: Array,
    num_steps: int,
    method: Literal["euler", "midpoint"] = "euler",
) -> tuple[Array, Array]:
    """Integrate an ODE from time zero to one for a batch of samples.

    ``velocity_function`` operates on one sample and one scalar time. The sampler
    applies it across the leading sample dimension and returns the terminal samples
    together with a trajectory whose first entry is ``initial_samples``.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if method not in {"euler", "midpoint"}:
        raise ValueError("method must be 'euler' or 'midpoint'")

    step_size = jnp.asarray(1 / num_steps, dtype=initial_samples.dtype)
    batched_velocity = jax.vmap(velocity_function, in_axes=(0, None))

    def step(samples: Array, step_index: Array) -> tuple[Array, Array]:
        time = step_index.astype(initial_samples.dtype) * step_size
        velocity = batched_velocity(samples, time)

        if method == "euler":
            next_samples = samples + step_size * velocity
        else:
            midpoint_samples = samples + 0.5 * step_size * velocity
            midpoint_velocity = batched_velocity(
                midpoint_samples, time + 0.5 * step_size
            )
            next_samples = samples + step_size * midpoint_velocity

        return next_samples, next_samples

    final_samples, path_without_initial = jax.lax.scan(
        step, initial_samples, jnp.arange(num_steps)
    )
    trajectory = jnp.concatenate((initial_samples[None, ...], path_without_initial))
    return final_samples, trajectory


def sample_euler(
    velocity_function: VelocityFunction, initial_samples: Array, num_steps: int
) -> tuple[Array, Array]:
    """Generate samples with fixed-step Euler integration."""
    return integrate_ode(velocity_function, initial_samples, num_steps, method="euler")


def sample_midpoint(
    velocity_function: VelocityFunction, initial_samples: Array, num_steps: int
) -> tuple[Array, Array]:
    """Generate samples with fixed-step midpoint integration."""
    return integrate_ode(
        velocity_function, initial_samples, num_steps, method="midpoint"
    )
