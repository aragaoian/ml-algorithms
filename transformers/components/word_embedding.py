import numpy as np
import numpy.typing as npt

from utils.activation_functions import ActivionFunctions


class WordEmbedding:
    def __init__(
        self,
        n_features: int,
        n_samples: int,
        hidden_size: int = 2,
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.n_samples = n_samples
        self.hidden_size = hidden_size

        rng = np.random.default_rng(seed)
        self.w = [
            rng.normal(0, 0.1, (self.n_features, hidden_size)),
            rng.normal(0, 0.1, (self.n_features, hidden_size)),
        ]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.hidden_size)]
        self.grad_w = [None for _ in range(self.hidden_size)]
        self.grad_b = [None for _ in range(self.hidden_size)]

        self.x = None
        self.h = None
        self.prob = None

    def forward_pass(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.x = x
        self.h = self.w[0].T @ self.x
        z = self.w[1] @ self.h + self.b[1]
        self.prob = ActivionFunctions.softmax(z)
        return self.w[0]

    def backward_pass(self, grad_out: npt.NDArray[np.float64]) -> None:
        self.reset_grads()

        # TODO
        # review backward pass

        # output layer
        grad_w1 = np.zeros_like(self.w[1])
        grad_b = np.zeros_like(self.b[1])
        for k in range(self.n_features):
            grad_w1[k] = (grad_out[k] * self.h).ravel()
            grad_b[k] = grad_out[k]

        # hidden layer
        upstream = np.zeros((self.hidden_size, 1))
        for i in range(self.hidden_size):
            upstream[i] = sum(
                grad_out[k] * self.w[1][k][i] for k in range(self.n_features)
            )

        # input layer
        grad_w0 = np.zeros_like(self.w[0])
        for i in range(self.hidden_size):
            grad_w0[:, i] = (upstream[i] * self.x).ravel()

        self.grad_w[0] = grad_w0
        self.grad_w[1] = grad_w1
        self.grad_b[1] = grad_b

    def reset_grads(self) -> None:
        self.grad_w = [None for _ in range(self.hidden_size)]
        self.grad_b = [None for _ in range(self.hidden_size)]

    def update_weights(self, l_rate: float) -> None:
        for i in range(self.hidden_size):
            self.w[i] -= l_rate * self.grad_w[i]
            self.b[i] -= l_rate * (self.grad_b[i] if self.grad_b[i] is not None else 0)

    def embeddings(self) -> npt.NDArray[np.float64]:
        return self.w[0]

    def formatted_embeddings(
        self,
        vocabulary: list[str],
    ) -> dict[str, npt.NDArray[np.float64]]:
        return {word: self.w[0][i] for i, word in enumerate(vocabulary)}
