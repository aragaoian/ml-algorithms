import numpy as np
import numpy.typing as npt

from transformers.components.cross_attention import CrossAttention
from transformers.components.feed_foward import FeedForward
from transformers.components.linear_layer import LinearLayer
from transformers.components.positional_encoder import PositionalEncoder
from transformers.components.self_attention import SelfAttention
from transformers.components.word_embedding import WordEmbedding
from utils.activation_functions import ActivionFunctions


class Decoder:
    def __init__(self, n_out: int, seed: int = 42) -> None:
        self.seed = seed
        self.probs = None
        self.n_out = n_out

        self.we = WordEmbedding(seed=self.seed + 4)
        self.pe = PositionalEncoder()
        self.self_att = SelfAttention(seed=self.seed + 5)
        self.cross_att = CrossAttention(seed=self.seed + 6)
        self.ff = FeedForward(seed=self.seed + 7)
        self.linear = LinearLayer(self.n_out, seed=seed + 8)

    def foward_pass(
        self,
        x: npt.NDArray[np.float64],
        encoder_z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        embeddings = self.we.forward_pass(x)  # (n, 2) -> for h = 2
        pos_encoded_embeddings = self.pe.generate(embeddings)  # (n, 2)

        z_self_att = self.self_att.foward_pass(
            pos_encoded_embeddings, masked_att=True
        )  # (n, 2)
        z_cross_att = self.cross_att.foward_pass(z_self_att, encoder_z)  # (n, 2)
        z_ff = self.ff.forward_pass(z_cross_att)  # (n, 2)
        z_linear = self.linear.forward_pass(z_ff)  # (n, vocab_size/n_out)
        self.probs = ActivionFunctions.softmax(
            z_linear, axis=1
        )  # (n, vocab_size/n_out)

        return self.probs

    def backward_pass(
        self,
        y: npt.NDArray[np.float64],
        l_rate: float,
    ) -> npt.NDArray[np.float64]:
        grad_probs = (self.probs - y) / self.probs.shape[0]
        grad_linear = self.linear.backward_pass(grad_probs)
        grad_ff = self.ff.backward_pass(grad_linear)
        grad_decoder, grad_encoder = self.cross_att.backward_pass(grad_ff)
        grad_self_att = self.self_att.backward_pass(grad_decoder)
        self.we.backward_pass(grad_self_att)

        self.we.update_weights(l_rate)
        self.self_att.update_weights(l_rate)
        self.cross_att.update_weights(l_rate)
        self.ff.update_weights(l_rate)
        self.linear.update_weights(l_rate)

        return grad_encoder
