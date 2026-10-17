import numpy as np
import numpy.typing as npt

from transformers.components.cross_attention import CrossAttention
from transformers.components.feed_foward import Feedfoward
from transformers.components.positional_encoder import PositionalEncoder
from transformers.components.self_attention import SelfAttention
from transformers.components.word_embedding import WordEmbedding


class Decoder:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.we = WordEmbedding(seed=self.seed + 4)
        self.pe = PositionalEncoder()
        self.self_att = SelfAttention(seed=self.seed + 5)
        self.cross_att = CrossAttention(seed=self.seed + 6)
        self.ff = Feedfoward(seed=self.seed + 7)

    def foward_pass(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        encoder_z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        embeddings = self.we.forward_pass(x)  # (n, 2) -> for h = 2
        pos_encoded_embeddings = self.pe.generate(embeddings)  # (n, 2)

        z_self_att = self.self_att.foward_pass(
            pos_encoded_embeddings, masked_att=True
        )  # (n, 2)
        z_cross_att = self.cross_att.foward_pass(z_self_att, encoder_z)  # (n, 2)
        z_ff = self.ff.forward_pass(z_cross_att)  # (n, 2)

        return z_ff

    def backward_pass(
        self,
        y: npt.NDArray[np.float64],
        l_rate: float,
    ) -> None:
        raise NotImplementedError
