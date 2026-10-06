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
        self.b = np.zeros((self.n_features, 1))
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
        self.a = ActivionFunctions.softmax(masked_qkt / self.key.shape[1], axis=1)
        return self.a @ self.values

    def backward_pass():
        # TODO
        # recieve dZ (next layer error)
        # update Wq, Wk, Wv
        # return dX (embedding layer)
        raise NotImplementedError

    def fit():
        raise NotImplementedError
