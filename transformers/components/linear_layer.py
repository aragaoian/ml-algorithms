import numpy as np
import numpy.typing as npt


class LinearLayer:
    def __init__(self, n_out: int, seed: int = 42):
        self.seed = seed
        self.n_out = n_out  # total vocab size
        self.n_in = None  # embedding size
        self.w = None
        self.b = None
        self.grad_w = None
        self.grad_b = None
        self.x = None

    def forward_pass(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.x = x
        _, self.n_in = x.shape

        if self.w is None:
            rng = np.random.default_rng(self.seed)
            self.w = rng.normal(0, 0.1, (self.n_in, self.n_out))
            self.b = np.zeros((self.n_out,))

        z = x @ self.w + self.b  # (n, n_out)
        return z

    def backward_pass(
        self, grad_out: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        self.grad_w = self.x.T @ grad_out  # (n_in, n_out)
        self.grad_b = np.sum(grad_out, axis=0)  # (n_out, )
        grad_z = grad_out @ self.w.T  # (n, n_in)
        return grad_z

    def reset_grads(self) -> None:
        self.grad_w = None
        self.grad_b = None

    def update_weights(self, l_rate: float) -> None:
        self.w -= l_rate * self.grad_w
        self.b -= l_rate * self.grad_b
