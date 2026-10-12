import jax.numpy as jnp
from jax import Array, random, tree_util

from nn_jax.activation import Tanh
from nn_jax.layer import Dense
from nn_jax.sequential import Sequential

from .time import sinusoidal_time_embedding


@tree_util.register_pytree_node_class
class TimeConditionedMLP:
    """An MLP that predicts from one feature vector and one scalar time."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int,
        time_embedding_size: int,
        key: Array,
    ):
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        if any(hidden_size <= 0 for hidden_size in hidden_sizes):
            raise ValueError("hidden_sizes must contain only positive integers")
        if time_embedding_size <= 0 or time_embedding_size % 2:
            raise ValueError("time_embedding_size must be a positive even integer")

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.time_embedding_size = time_embedding_size
        layer_sizes = (
            input_size + time_embedding_size,
            *hidden_sizes,
            output_size,
        )
        layer_keys = random.split(key, len(layer_sizes) - 1)
        modules = []
        for layer_index, (in_size, out_size, layer_key) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], layer_keys, strict=True)
        ):
            modules.append(Dense(in_size, out_size, layer_key))
            if layer_index < len(layer_keys) - 1:
                modules.append(Tanh())
        self.network = Sequential(modules)

    def predict(self, x: Array, time: Array) -> Array:
        """Predict an output for one input vector at one scalar time."""
        embedding = sinusoidal_time_embedding(time, self.time_embedding_size)
        return self.network.forward(jnp.concatenate((x, embedding)))

    def tree_flatten(self):
        children = (self.network,)
        aux_data = (
            self.input_size,
            self.hidden_sizes,
            self.output_size,
            self.time_embedding_size,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        input_size, hidden_sizes, output_size, time_embedding_size = aux_data
        (network,) = children
        instance = cls.__new__(cls)
        instance.input_size = input_size
        instance.hidden_sizes = hidden_sizes
        instance.output_size = output_size
        instance.time_embedding_size = time_embedding_size
        instance.network = network
        return instance
