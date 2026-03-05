from jax import tree_util

from nn_jax.activation_function.activation_function import ActivationFunction
from nn_jax.activation_function.relu import ReLU
from nn_jax.activation_function.softmax import Softmax

__all__ = [ActivationFunction, ReLU, Softmax]

for activation_function in __all__:
    tree_util.register_pytree_node(
        activation_function,
        activation_function._tree_flatten,
        activation_function._tree_unflatten,
    )
