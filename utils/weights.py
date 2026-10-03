import numpy as np


class Weights:
    """
    Weights are best initialized as arrays since each element corresponds to the weight of a specific Xi.
    If represented as a column vector (n x 1), proper reshaping is required to ensure a correct dot product.
    """

    def __init__(self, n_features: int):
        self.lim = 1 / np.sqrt(n_features)
        self.n_features = n_features
        self.seed = 42
        np.random.seed(self.seed)

    def random_rand(self):
        return np.random.rand(self.n_features, 1)

    def random_normal(self):
        return np.random.normal(-self.lim, self.lim, (self.n_features, 1))

    def random_uniform(self):
        return np.random.uniform(-self.lim, self.lim, (self.n_features, 1))
