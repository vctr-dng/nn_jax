from collections.abc import Iterable

from jax import tree_util
from jax.typing import ArrayLike

from nn_jax.module import Module


@tree_util.register_pytree_node_class
class Sequential(Module):
    def __init__(self, modules: Iterable[Module]):
        self.modules: tuple[Module, ...] = tuple(modules)

    def forward(self, x: ArrayLike) -> ArrayLike:
        for module in self.modules:
            x = module.forward(x)
        return x

    def tree_flatten(self):
        children = self.modules
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(tuple(children))
