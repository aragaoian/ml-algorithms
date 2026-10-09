import numpy as np
import numpy.typing as npt

from utils.activation_functions import ActivionFunctions


class FeedFoward:
    def __init__(
        self,
        X: npt.NDArray[np.float16],
        y: npt.NDArray[np.float16],
        hidden_size: int = 2,
        n_iter: int = 5,
        l_rate: float = 0.001,
        seed: int = 42,
    ):
        self.X = X
        self.y = y
        self.n_features = X.shape[0]
        self.n_samples = X.shape[1]
        self.hidden_size = hidden_size
        self.n_iter = n_iter
        self.l_rate = l_rate
        rng = np.random.default_rng(seed)
        self.w = [
            rng.normal(0, 0.1, (self.n_features, hidden_size)),
            rng.normal(0, 0.1, (self.n_features, hidden_size)),
        ]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.hidden_size)]

    def forward_pass(self, x):
        # NOTE
        # non-linearity goes between two W layers
        f = self.w[0].T @ x + self.b[0]  # (n, 2)
        h = ActivionFunctions.relu(f)  # (n, 2)
        z = self.w[1] @ h + self.b[1]  # (n, 2)
        prob = ActivionFunctions.softmax(z)
        return f, h, prob

    def backward_pass(self, x, f, h, grad_out):
        # output layer
        grad_w1 = grad_out @ h
        grad_b1 = np.sum(grad_out, axis=0)

        # hidden layer
        grad_h = grad_out @ self.w[1].T

        # through the relu
        grad_f = grad_h * (f > 0)  # should be element-wise

        # input layer
        grad_w0 = grad_f @ x
        grad_b0 = np.sum(grad_f, axis=0)

        grad_z = grad_f @ self.w[0]

        return grad_z, grad_w0, grad_b0, grad_w1, grad_b1
