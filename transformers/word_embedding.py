import numpy as np
import numpy.typing as npt

from utils.activation_functions import ActivionFunctions


class WordEmbedding:
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
        self.b = np.zeros((self.n_features, 1))

    def forward_pass(self, x):
        h = self.w[0].T @ x
        z = self.w[1] @ h + self.b
        prob = ActivionFunctions.softmax(z)
        return h, prob

    def backward_pass(self, x, t, h, prob):
        # output layerp[]
        errors = prob - t
        grad_w1 = np.zeros_like(self.w[1])
        grad_b = np.zeros_like(self.b)
        for k in range(self.n_features):
            grad_w1[k] = (errors[k] * h).ravel()
            grad_b[k] = errors[k]

        # hidden layer
        upstream = np.zeros((self.hidden_size, 1))
        for i in range(self.hidden_size):
            upstream[i] = sum(
                errors[k] * self.w[1][k][i] for k in range(self.n_features)
            )

        # input layer
        grad_w0 = np.zeros_like(self.w[0])
        for i in range(self.hidden_size):
            grad_w0[:, i] = (upstream[i] * x).ravel()

        return grad_w0, grad_w1, grad_b

    def fit(self):
        for epoch in range(self.n_iter):
            total_loss = 0.0
            for s in range(self.n_samples):
                x = self.X[:, s].reshape(-1, 1)
                t = self.y[:, s].reshape(-1, 1)

                h, prob = self.forward_pass(x)
                total_loss += -np.sum(t * np.log(prob + 1e-12))

                grad_w0, grad_w1, grad_b = self.backward_pass(x, t, h, prob)

                self.w[0] -= self.l_rate * grad_w0
                self.w[1] -= self.l_rate * grad_w1
                self.b -= self.l_rate * grad_b

    def embeddings(self):
        return self.w[0]

    def formatted_embeddings(
        self,
        vocabulary: list[str],
    ) -> dict[str, np.float16]:
        return {word: self.w[0][i] for i, word in enumerate(vocabulary)}
