from abc import ABC, abstractmethod


class Optimizer[ModelT](ABC):
    @abstractmethod
    def __call__(self, model: ModelT, grads: ModelT) -> ModelT:
        pass
