import numpy as np
import numpy.typing as npt

from utils.activation_functions import ActivionFunctions


class CrossAttention:
    def __init__(
        self,
        n_features: int,
        n_embeddings: int,
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.n_embeddings = n_embeddings
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.wq = rng.normal(0, 0.1, (n_embeddings, n_embeddings))
        self.query = None
        self.key = None
        self.values = None
        self.a = None

    def foward_pass(
        self,
        x: npt.NDArray[np.float64],
        k: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        self.query = x @ self.wq
        self.key = x @ k
        self.values = x @ v

        qkt = np.matmul(self.query, self.key.T)
        scores = qkt / np.sqrt(self.key.shape[1])
        self.a = ActivionFunctions.softmax(scores, axis=1)
        z = self.a @ self.values
        return z

    def backward_pass(
        self,
        x: npt.NDArray[np.float64],
        grad_out: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

        grad_wq = x.T @ grad_q
        grad_wk = x.T @ grad_k
        grad_wv = x.T @ grad_v

        return grad_wq, grad_wk, grad_wv
