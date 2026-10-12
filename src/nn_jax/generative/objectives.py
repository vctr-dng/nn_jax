from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from .paths import conditional_flow_matching_path


def conditional_flow_matching_loss(
    velocity_function: Callable[[Array, Array], Array],
    noise: Array,
    data: Array,
    time: Array,
    sigma_min: float = 0.0,
) -> Array:
    """Return the squared error against a conditional flow-matching target."""
    sample, velocity_target = conditional_flow_matching_path(
        noise, data, time, sigma_min
    )
    predicted_velocity = velocity_function(sample, time)
    return jnp.mean((predicted_velocity - velocity_target) ** 2)
