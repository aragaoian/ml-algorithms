import numpy as np
import numpy.typing as npt

from transformers.components.feed_foward import Feedfoward
from transformers.components.positional_encoder import PositionalEncoder
from transformers.components.self_attention import SelfAttention
from transformers.components.word_embedding import WordEmbedding


class Encoder:
    def __init__(self, n_features: int, n_samples: int, seed: int) -> None:
        self.n_features = n_features
        self.n_samples = n_samples
        self.seed = seed
        self.we = WordEmbedding(n_features, n_samples, seed=seed + 1)
        self.pe = PositionalEncoder()
        self.self_att = SelfAttention(n_features, n_samples, seed=seed + 2)
        self.ff = Feedfoward(n_features, n_samples, seed=seed + 3)

    def foward_pass(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        embeddings = self.we.forward_pass(x)
        pos_encoded_embeddings = self.pe.generate(embeddings)

        z_att = self.self_att.foward_pass(pos_encoded_embeddings, masked_att=False)
        z_ff = self.ff.forward_pass(z_att)

        return z_ff

    def backward_pass(
        self, decoder_grad: npt.NDArray[np.float64], l_rate: float
    ) -> None:
        grad_z = self.ff.backward_pass(decoder_grad)
        grad_x = self.self_att.backward_pass(grad_z)
        self.we.backward_pass(grad_x)

        self.we.update_weights(l_rate)
        self.self_att.update_weights(l_rate)
        self.ff.update_weights(l_rate)


import numpy.typing as npt
