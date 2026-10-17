import numpy as np
import numpy.typing as npt

from utils.activation_functions import ActivionFunctions


class CrossAttention:
    def __init__(
        self,
        seed: int = 42,
    ) -> None:
        self.n_features: int | None = None
        self.n_encodings: int | None = None
        self.seed = seed
        self.w = None
        self.grad_w = {
            "grad_wq": None,
            "grad_wk": None,
            "grad_wv": None,
        }
        self.grad_x = None
        self.grad_z = None

        self.query = None
        self.key = None
        self.values = None
        self.a = None
        self.x = None
        self.z = None

    def foward_pass(
        self,
        x: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        self.n_features, self.n_encodings = x.shape

        if self.w is None:
            rng = np.random.default_rng(self.seed)
            self.w = {
                name: rng.normal(0, 0.1, (self.n_encodings, self.n_encodings))
                for name in ("q", "k", "v")
            }

        self.x = x
        self.z = z

        self.query = x @ self.w["q"]
        self.key = z @ self.w["k"]
        self.values = z @ self.w["v"]

        qkt = np.matmul(self.query, self.key.T)
        scores = qkt / np.sqrt(self.key.shape[1])
        self.a = ActivionFunctions.softmax(scores, axis=1)
        z = self.a @ self.values
        return z

    def backward_pass(
        self,
        grad_out: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self.reset_grads()

        # values
        grad_v = self.a.T @ grad_out  # (n, 2)

        # attention W
        grad_att = grad_out @ self.values.T  # (n, 2)

        # raw scores
        grad_rs = self.a * (
            grad_att - np.sum(grad_att * self.a, axis=1, keepdims=True)
        )  # (n, n)

        # queries and keys
        d_key = self.key.shape[1]
        grad_q = (grad_rs @ self.key) / np.sqrt(d_key)  # (n, 2)
        grad_k = (grad_rs.T @ self.query) / np.sqrt(d_key)  # (n, 2)

        self.grad_w["grad_wq"] = self.x.T @ grad_q
        self.grad_w["grad_wk"] = self.z.T @ grad_k
        self.grad_w["grad_wv"] = self.z.T @ grad_v

        self.grad_x = grad_q @ self.w["q"].T
        self.grad_z = grad_k @ self.w["k"].T + grad_v @ self.w["v"].T

        return self.grad_x, self.grad_z

    def reset_grads(self) -> None:
        self.grad_w = {
            "grad_wq": None,
            "grad_wk": None,
            "grad_wv": None,
        }
        self.grad_x = None
        self.grad_z = None

    def update_weights(self, l_rate: float) -> None:
        self.w["q"] -= l_rate * self.grad_w["grad_wq"]
        self.w["k"] -= l_rate * self.grad_w["grad_wk"]
        self.w["v"] -= l_rate * self.grad_w["grad_wv"]
