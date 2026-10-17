import numpy as np
import numpy.typing as npt


class WordEmbedding:
    def __init__(
        self,
        hidden_size: int = 2,
        seed: int = 42,
    ) -> None:
        self.n_features: int | None = None
        self.hidden_size = hidden_size
        self.seed = seed
        self.w = None
        self.grad_w = None

        self.x = None
        self.h = None
        self.prob = None

    def forward_pass(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.n_features, _ = x.shape  # vocab size, _

        if self.w is None:
            rng = np.random.default_rng(self.seed)
            self.w = [
                rng.normal(0, 0.1, (self.n_features, self.hidden_size)),
            ]
            self.grad_w = [None]

        self.x = x
        self.h = self.x.T @ self.w[0]
        # NOTE
        # Head was only necessary on the old architecture.
        # When working a full encoder-decoder, head would
        # a second, unrelated training signal (weights that
        # do not take any part in the training).
        return self.h

    def backward_pass(self, grad_out: npt.NDArray[np.float64]) -> None:
        self.reset_grads()

        # input layer
        self.grad_w[0] = self.x @ grad_out  # (vocab, n) @ (n, 2) -> (vocab, 2)

    def reset_grads(self) -> None:
        self.grad_w = [None]

    def update_weights(self, l_rate: float) -> None:
        self.w[0] -= l_rate * self.grad_w[0]

    def formatted_embeddings(
        self,
        vocabulary: list[str],
    ) -> dict[str, npt.NDArray[np.float64]]:
        return {word: self.w[0][i] for i, word in enumerate(vocabulary)}
