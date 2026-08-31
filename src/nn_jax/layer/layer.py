from ..module import Module


class Layer(Module):
    def __init__(self, in_size: int | tuple[int, ...], out_size: int | tuple[int, ...]):
        self.in_size: int | tuple[int, ...] = in_size
        self.out_size: int | tuple[int, ...] = out_size
