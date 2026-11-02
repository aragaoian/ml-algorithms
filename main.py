import numpy as np

from transformers.data.words import PAIRS, PT, encode_pairs
from transformers.transformer import Transformer

train, test = encode_pairs(PAIRS)

t = Transformer(
    decoder_n_out=len(PT),
    n_epochs=5000,
    l_rate=0.01,
    threshold=-1,
)
loss = t.fit(train)


X_test = test[0][0]
ohe_matrix, predicted_tokens = t.predict(X_test)

print("Predicted:", [PT[token] for token in predicted_tokens])

# TODO
# fix model implemention
# Predicted: ['Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu', 'Eu']

# I fixed some stuff but the model still underfits
