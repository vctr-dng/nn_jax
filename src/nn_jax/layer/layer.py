from abc import abstractmethod

import jax.numpy as jnp
from jax import Array

from ..module import Module


class Layer(Module):
    has_params = True

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: Array | None = None,
        weights: Array | None = None,
        bias: Array | None = None,
    ):
        self.in_size: int = in_size
        self.out_size: int = out_size

        if not self.has_params:
            if key is not None or weights is not None or bias is not None:
                raise ValueError(
                    "Parameterless layers cannot accept key, weights, or bias"
                )
            return

        if key is not None and weights is not None:
            raise ValueError("Key and weights cannot be provided together")

        if weights is None:
            if key is None:
                raise ValueError("Key is required if weights are not provided")
            self._init_weights(key)
        else:
            if weights.shape != self._weights_shape:
                raise ValueError(
                    f"Weights shape must be {self._weights_shape}, got {weights.shape}"
                )
            self.weights = weights

        if bias is not None:
            if bias.shape != (self.out_size,):
                raise ValueError(
                    f"Bias shape must be ({self.out_size},), got {bias.shape}"
                )
            self.bias = bias
        else:
            self.bias = jnp.zeros((self.out_size,))

    @abstractmethod
    def _init_weights(self, key: Array):
        pass

    @property
    @abstractmethod
    def _weights_shape(self) -> tuple[int, ...]:
        pass

    @classmethod
    def from_parameters(
        cls,
        in_size: int,
        out_size: int,
        weights: Array,
        bias: Array,
    ) -> "Layer":
        return cls(in_size, out_size, weights=weights, bias=bias)

    @abstractmethod
    def tree_flatten(self) -> tuple[tuple, tuple]:
        pass

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux_data, children) -> "Layer":
        pass
