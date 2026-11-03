import numpy as np
import numpy.typing as npt


class Norm:
    def __init__(self):
        pass

    @staticmethod
    def layer_norm(x: npt.NDArray) -> npt.NDArray:
        pass
        # NOTE
        # The way I am implemeting this tranformers
        # there is not need to use layer norm. Gradients
        # are already to small (1e-3) for d=2
        # If i want to test layer norm os RSMNorm, I should
        # increase d to, at least, 8.
