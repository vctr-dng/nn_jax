import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .loss import Loss


class CategoricalCrossEntropy(Loss):
    CLIPPING_VALUE = 1e-8

    def __init__(
        self, min_val: float = CLIPPING_VALUE, max_val: float = 1 - CLIPPING_VALUE
    ):
        self.min_val = min_val
        self.max_val = max_val

    def calculate_loss(self, pred: ArrayLike, target: ArrayLike) -> Array:
        # target must be a one-hot encoded array

        pred_clipped = jnp.clip(pred, self.min_val, self.max_val)

        negative_log_likelihoods = -jnp.sum(target * jnp.log(pred_clipped), axis=-1)

        return jnp.mean(negative_log_likelihoods)
