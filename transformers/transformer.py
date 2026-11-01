import random

import numpy as np
import numpy.typing as npt
from rich.progress import track

from transformers.data.words import EOS_TAG, PT_IDX, ohe_columns
from transformers.decoder import Decoder
from transformers.encoder import Encoder

MAX_LOOP_LEN = 10


class Transformer:
    def __init__(
        self,
        decoder_n_out: int,
        n_epochs: int = 200,
        l_rate: float = 0.01,
        threshold: float = 0.01,
        seed: int = 42,
    ):
        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.threshold = threshold
        self.seed = seed
        self.encoder = Encoder(seed)
        self.decoder = Decoder(decoder_n_out, seed)
        self.loss_history = None

        random.seed(seed)

    def _decoder_input(self) -> npt.NDArray:
        eos_token = [EOS_TAG]
        return ohe_columns(eos_token, PT_IDX)

    def fit(self, pairs: list[tuple[npt.NDArray, npt.NDArray, npt.NDArray]]) -> None:
        self.loss_history = []

        for _ in track(range(self.n_epochs), description="Epochs"):
            shuffled_pairs = random.sample(pairs, len(pairs))
            epoch_loss = []

            for X_encoder, X_decoder, y in shuffled_pairs:
                encoder_z = self.encoder.forward_pass(X_encoder)
                _, loss = self.decoder.forward_pass(X_decoder, encoder_z, y)

                epoch_loss.append(loss)

                if np.mean(epoch_loss) < self.threshold:
                    break

                decoder_grad = self.decoder.backward_pass(y, self.l_rate)
                self.encoder.backward_pass(decoder_grad, self.l_rate)

            self.loss_history.append(np.mean(epoch_loss))

        return self.loss_history

    def predict(self, X_encoder: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        encoder_z = self.encoder.forward_pass(X_encoder)
        X_decoder = self._decoder_input()

        for _ in range(MAX_LOOP_LEN):
            probs, _ = self.decoder.forward_pass(X_decoder, encoder_z)
            predict_token = np.argmax(probs[-1])
            if predict_token == PT_IDX[EOS_TAG]:
                return probs
            X_decoder = np.append(X_decoder, predict_token, axis=0)

        return X_decoder
