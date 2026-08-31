import jax

from ..sequential import Sequential
from .optimizer import Optimizer


class StochasticGradientDescent(Optimizer[Sequential]):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate: float = learning_rate

    def __call__(self, model: Sequential, grads: Sequential) -> Sequential:
        return jax.tree.map(
            lambda param, grad: param - self.learning_rate * grad,
            model,
            grads,
        )
