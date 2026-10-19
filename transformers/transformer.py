from transformers.decoder import Decoder
from transformers.encoder import Encoder


class Transformer:
    def __init__(
        self, n_out: int, n_epochs: int = 50, l_rate: float = 0.001, seed: int = 42
    ):
        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.seed = seed
        self.encoder = Encoder(seed)
        self.decoder = Decoder(n_out, seed)

    def fit(self, X, y):
        for epoch in range(self.n_epochs):
            encoder_z = self.encoder.foward_pass(X)
            self.decoder.foward_pass(X, encoder_z)

            decoder_grad = self.decoder.backward_pass(y, self.l_rate)
            self.encoder.backward_pass(decoder_grad, self.l_rate)

            # TODO
            # implement loss threshold
