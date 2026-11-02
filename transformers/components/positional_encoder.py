import numpy as np
import numpy.typing as npt


class PositionalEncoder:
    def __init__(self) -> None:
        pass

    def generate(self, embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        sequence_length, hidden_size = embeddings.shape
        positions = np.arange(sequence_length)[:, np.newaxis]
        dimensions = np.arange(hidden_size)[np.newaxis, :]
        angle_rates = 1 / np.power(
            10000, (2 * (dimensions // 2)) / hidden_size
        )
        angles = positions * angle_rates

        positional_encoding = np.empty_like(embeddings)
        positional_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])
        return embeddings + positional_encoding
