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
        self.n_samples = X.shape[1]
        self.n_iter = n_iter
        self.l_rate = l_rate
        rng = np.random.default_rng(seed)
        self.w = {
            "q": rng.normal(0, 0.1, (self.n_samples, self.n_samples)),
            "k": rng.normal(0, 0.1, (self.n_samples, self.n_samples)),
            "v": rng.normal(0, 0.1, (self.n_samples, self.n_samples)),
        }
        self.b = np.zeros((self.n_features, 1))

    def fit(self):
        query = self.X @ self.w["q"]
        key = self.X @ self.w["k"]
        values = self.X @ self.w["v"]

        a = ActivionFunctions.softmax(np.matmul(query, key.T)) / np.sqrt(key.shape[1])
        z = a @ values
