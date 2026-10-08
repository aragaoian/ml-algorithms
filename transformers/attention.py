import numpy as np
import numpy.typing as npt

from utils.activation_functions import ActivionFunctions


class Attention:
    def __init__(
        self,
        X: npt.NDArray[np.float16],
        y: npt.NDArray[np.float16],
        n_iter: int = 5,
        l_rate: float = 0.001,
        seed: int = 42,
    ):
        self.X = X
        self.y = y
        self.n_features = X.shape[0]
        self.n_embeddings = X.shape[1]
        self.n_iter = n_iter
        self.l_rate = l_rate
        rng = np.random.default_rng(seed)
        self.w = {
            "q": rng.normal(0, 0.1, (self.n_embeddings, self.n_embeddings)),
            "k": rng.normal(0, 0.1, (self.n_embeddings, self.n_embeddings)),
            "v": rng.normal(0, 0.1, (self.n_embeddings, self.n_embeddings)),
        }
        self.query = None
        self.key = None
        self.values = None
        self.a = None

    def foward_pass(self):
        """ "Return attention head result"""
        self.query = self.X @ self.w["q"]
        self.key = self.X @ self.w["k"]
        self.values = self.X @ self.w["v"]

        qkt = np.matmul(self.query, self.key.T)
        mask = np.triu(np.ones_like(qkt), k=1) * -1e9
        masked_qkt = qkt + mask
        scores = masked_qkt / np.sqrt(self.key.shape[1])
        self.a = ActivionFunctions.softmax(scores, axis=1)
        return self.a @ self.values

    def backward_pass(self, dZ):
        # Values
        dV = self.a.T @ dZ  # (n, 2)

        # Attention W
        dA = dZ @ self.values.T  # (n, 2)

        # Raw Scores (S)
        dS = self.a * (dA - np.sum(dA * self.a, axis=1, keepdims=True))  # (n, n)

        # Queries and Keys
        d = self.key.shape[1]
        dQ = (dS @ self.key) / np.sqrt(d)  # (n, 2)
        dK = (dS.T @ self.query) / np.sqrt(d)  # (n, 2)

        return {
            "grad_wq": self.X.T @ dQ,
            "grad_wk": self.X.T @ dK,
            "grad_wv": self.X.T @ dV,
        }
