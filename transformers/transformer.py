import numpy as np
import numpy.typing as npt
from rich.progress import track

from transformers.decoder import Decoder
from transformers.encoder import Encoder


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

    def fit(
        self,
        X_encoder: npt.NDArray[np.float64],
        X_decoder: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> None:
        epochs_loss = []
        for epoch in track(range(self.n_epochs), description="Epochs"):
            encoder_z = self.encoder.forward_pass(X_encoder)
            _, loss = self.decoder.forward_pass(X_decoder, encoder_z, y)

            if loss < self.threshold:
                break

            epochs_loss.append(loss)

            decoder_grad = self.decoder.backward_pass(y, self.l_rate)
            self.encoder.backward_pass(decoder_grad, self.l_rate)

    def predict(
        self, X_encoder: npt.NDArray[np.float64], X_decoder: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        encoder_z = self.encoder.forward_pass(X_encoder)
        # TODO
        # add loop for the decoder
        probs, _ = self.decoder.forward_pass(X_decoder, encoder_z)
        return probs
