import numpy as np
import numpy.typing as npt


class PositionalEncoder:
    def __init__(self) -> None:
        pass

    def generate(self, embeddings: npt.NDArray[np.float16]) -> npt.NDArray[np.float16]:
        n_features, n_samples = embeddings.shape
        sin_cos_arr = np.zeros_like(embeddings)
        for pos in range(n_features):
            for s in range(n_samples):
                sin_cos_arr[pos][s] = np.sin(pos)
                sin_cos_arr[pos][s] = np.cos(s)

        return embeddings + sin_cos_arr
