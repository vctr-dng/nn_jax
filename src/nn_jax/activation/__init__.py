from jax import tree_util

from .activation import Activation
from .relu import ReLU
from .softmax import Softmax
from .tanh import Tanh

__all__ = ["Activation", "ReLU", "Softmax", "Tanh"]

for activation_function in __all__:
    tree_util.register_pytree_node(
        activation_function,
        activation_function._tree_flatten,
        activation_function._tree_unflatten,
    )
