import numpy as np
import numpy.typing as npt


class Decoder:
    def __init__(self, n_features: int, n_samples: int, seed: int) -> None:
        pass

    def foward_pass(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def backward_pass(
        self, output_grad: npt.NDArray[np.float64], l_rate: float
    ) -> None:
        raise NotImplementedError
