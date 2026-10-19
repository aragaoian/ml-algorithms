import numpy as np
import numpy.typing as npt

from transformers.components.feed_foward import FeedForward
from transformers.components.positional_encoder import PositionalEncoder
from transformers.components.self_attention import SelfAttention
from transformers.components.word_embedding import WordEmbedding


class Encoder:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.we = WordEmbedding(seed=self.seed + 1)
        self.pe = PositionalEncoder()
        self.self_att = SelfAttention(seed=self.seed + 2)
        self.ff = FeedForward(seed=self.seed + 3)

    def foward_pass(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        embeddings = self.we.forward_pass(x)  # (n, 2) -> for h = 2
        pos_encoded_embeddings = self.pe.generate(embeddings)  # (n, 2)

        z_att = self.self_att.foward_pass(
            pos_encoded_embeddings, masked_att=False
        )  # (n, 2)
        z_ff = self.ff.forward_pass(z_att)  # (n, 2)

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
