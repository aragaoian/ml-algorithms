import numpy as np
import numpy.typing as npt


class PositionalEncoder:
    def __init__(self) -> None:
        pass

    def generate(self, embeddings: npt.NDArray[np.float16]) -> npt.NDArray[np.float16]:
        n_features, _ = embeddings.shape[0]
        sin_cos_arr = np.zeros_like(embeddings)
        for pos in range(n_features):
            sin_cos_arr[pos][0] = np.sin(pos)
            sin_cos_arr[pos][1] = np.cos(pos)

        return embeddings + sin_cos_arr
