from jax import Array
from jax.typing import ArrayLike

from nn_jax.module import Module


class Sequential(Module):
    def __init__(self, modules: list[Module]):
        super().__init__()
        self.modules: list[Module] = modules
        self.n_parameters: list[int] = [len(module.parameters) for module in modules]

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        for module in self.modules:
            inputs = module.forward(inputs)
        return inputs

    def backward(self, out_grad: ArrayLike) -> ArrayLike:
        in_grad = out_grad
        for module in reversed(self.modules):
            in_grad = module.backward(in_grad)
        return in_grad

    @property
    def parameters(self) -> list[Array]:
        parameters: list[Array] = []
        for module in self.modules:
            parameters.extend(module.parameters)
        return parameters

    @parameters.setter
    def parameters(self, new_parameters: list[Array]):
        start_index = 0
        for i in range(len(self.modules)):
            n = self.n_parameters[i]
            updated_parameters = new_parameters[start_index : start_index + n]
            self.modules[i].parameters = updated_parameters
            start_index += n

    @property
    def gradients(self) -> list[Array]:
        gradients: list[Array] = []
        for module in self.modules:
            gradients.extend(module.gradients)
        return gradients

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()
