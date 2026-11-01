import numpy as np

from transformers.data.words import PAIRS, PT, encode_pairs
from transformers.transformer import Transformer

train, test = encode_pairs(PAIRS)

t = Transformer(decoder_n_out=len(PT))
loss = t.fit(train)

print(loss)
