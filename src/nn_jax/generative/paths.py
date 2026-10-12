import jax.numpy as jnp
from jax import Array


def conditional_flow_matching_path(
    noise: Array,
    data: Array,
    time: Array,
    sigma_min: float = 0.0,
) -> tuple[Array, Array]:
    """Return a straight conditional path sample and its velocity target."""
    if not 0 <= sigma_min < 1:
        raise ValueError("sigma_min must be in [0, 1)")

    time = jnp.asarray(time)
    time = time.reshape(time.shape + (1,) * (noise.ndim - time.ndim))
    noise_coefficient = 1 - (1 - sigma_min) * time
    sample = noise_coefficient * noise + time * data
    velocity = data - (1 - sigma_min) * noise
    return sample, velocity
