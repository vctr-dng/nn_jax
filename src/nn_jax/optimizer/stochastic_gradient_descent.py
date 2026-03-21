from jax import Array

from .optimizer import Optimizer
from nn_jax.module import Module


class StochasticGradientDescent(Optimizer):
    def __init__(self, module: Module, learning_rate: float = 0.01):
        super().__init__(module)
        self.learning_rate: float = learning_rate

    def step(self):
        parameters = self.module.parameters
        gradients = self.module.gradients

        new_parameters = list[Array]()
        for i in range(len(parameters)):
            new_parameters.append(parameters[i] - self.learning_rate * gradients[i])
        self.module.parameters = new_parameters
