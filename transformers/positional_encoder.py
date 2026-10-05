import numpy as np
import numpy.typing as npt


class PositionalEncoder:
    def __init__(self, embeddings: npt.NDArray[np.float16]):
        self.embeddings = embeddings
        self.n_features = self.embeddings.shape[0]

    def generate(self) -> npt.NDArray[np.float16]:
        sin_cos_arr = np.zeros_like(self.embeddings)
        for pos in range(self.n_features):
            sin_cos_arr[pos][0] = np.sin(pos)
            sin_cos_arr[pos][1] = np.cos(pos)

        return self.embeddings + sin_cos_arr
