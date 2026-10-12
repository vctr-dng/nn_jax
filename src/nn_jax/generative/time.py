import jax.numpy as jnp
from jax import Array


def sinusoidal_time_embedding(
    time: Array,
    embedding_size: int,
    max_period: float = 10_000.0,
) -> Array:
    """Encode scalar times with paired sine and cosine frequencies."""
    if embedding_size <= 0 or embedding_size % 2:
        raise ValueError("embedding_size must be a positive even integer")
    if max_period <= 0:
        raise ValueError("max_period must be positive")

    half_size = embedding_size // 2
    frequencies = jnp.exp(-jnp.log(max_period) * jnp.arange(half_size) / half_size)
    angles = jnp.asarray(time)[..., None] * frequencies
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)
