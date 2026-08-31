from nn_jax.module import Module


class Activation(Module):
    def __init__(self):
        super().__init__()

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()
