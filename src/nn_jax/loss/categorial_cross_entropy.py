import jax.numpy as jnp
from jax import Array

from .loss import Loss


class CategorialCrossEntropy(Loss):
    def __init__(self, min_val: float = 1e-8, max_val: float = 1 - 1e-8):
        self.min_val = min_val
        self.max_val = max_val

    def calculate_loss(self, y_pred: Array, y_true: Array) -> Array:
        y_pred_clipped = jnp.clip(y_pred, self.min_val, self.max_val)

        if len(y_true.shape) == 1:
            calculated_confidences = y_pred_clipped[range(y_true.shape[0]), y_true]
        else:
            calculated_confidences = jnp.sum(y_true * y_pred_clipped, axis=1)

        negative_log_likelihoods = -jnp.log(calculated_confidences)
        return negative_log_likelihoods
