import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .activation import Activation


class Softmax(Activation):
    def forward(self, inputs: ArrayLike) -> Array:
        super().forward(inputs)
        exps = jnp.exp(inputs - jnp.max(inputs, axis=0, keepdims=True))
        self.outputs = exps / jnp.sum(exps, axis=0, keepdims=True)
        return self.outputs

    def derivative(self, inputs: ArrayLike) -> Array:
        # Softmax's Jacobian is not diagonal (∂y_i/∂x_j = y_i(δ_ij - y_j)) so it has no elementwise derivative.
        # See backward() instead.
        raise NotImplementedError(
            "Softmax has no elementwise derivative, use backward() directly."
        )

    def backward(self, out_grad: ArrayLike) -> Array:
        y = self.outputs
        return y * (out_grad - jnp.sum(out_grad * y, axis=0, keepdims=True))
