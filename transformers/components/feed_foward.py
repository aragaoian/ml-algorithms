import numpy as np
import numpy.typing as npt

from utils.activation_functions import ActivionFunctions


class Feedfoward:
    def __init__(
        self,
        layers: int = 2,
        hidden_size: int = 8,
        seed: int = 42,
    ) -> None:
        self.n_features: int | None = None
        self.n_encodings: int | None = None
        self.layers = layers
        self.hidden_size = hidden_size
        self.seed = seed
        self.w = None
        self.b = None
        self.grad_w = [None, None]
        self.grad_b = [None, None]

        self.x = None
        self.pre_relu = None
        self.h = None

    def forward_pass(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.n_features, self.n_encodings = x.shape
        if self.w is None:
            rng = np.random.default_rng(self.seed)
            self.w = [
                rng.normal(0, 0.1, (self.n_encodings, self.hidden_size)),
                rng.normal(0, 0.1, (self.hidden_size, self.n_encodings)),
            ]
            self.b = [
                np.zeros((1, self.hidden_size)),
                np.zeros((1, self.n_encodings)),
            ]
        self.x = x
        # NOTE
        # non-linearity goes between two W layers
        self.pre_relu = self.x @ self.w[0] + self.b[0]
        self.h = ActivionFunctions.relu(self.pre_relu)
        z = self.h @ self.w[1] + self.b[1]
        return z  # (n, e) @ (h, e) -> (n, e)

    def backward_pass(
        self, grad_out: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        self.reset_grads()

        # output layer
        self.grad_w[1] = self.h.T @ grad_out
        self.grad_b[1] = np.sum(grad_out, axis=0)

        # hidden layer
        grad_h = grad_out @ self.w[1].T

        # through the relu
        grad_f = grad_h * (self.pre_relu > 0)  # should be element-wise

        # input layer
        self.grad_w[0] = self.x.T @ grad_f
        self.grad_b[0] = np.sum(grad_f, axis=0)

        grad_z = grad_f @ self.w[0].T

        return grad_z

    def reset_grads(self) -> None:
        self.grad_w = [None, None]
        self.grad_b = [None, None]

    def update_weights(self, l_rate: float) -> None:
        for i in range(self.layers):
            self.w[i] -= l_rate * self.grad_w[i]
            self.b[i] -= l_rate * self.grad_b[i]
