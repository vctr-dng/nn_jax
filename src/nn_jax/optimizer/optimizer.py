from abc import abstractmethod

from nn_jax import Module


class Optimizer:
    def __init__(self, module: Module):
        self.module: Module = module

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        self.module.zero_grad()
