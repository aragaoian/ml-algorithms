from transformers.data.words import EN, PAIRS, PT, encode_pairs
from transformers.transformer import Transformer

train, test = encode_pairs(PAIRS)

t = Transformer(
    decoder_n_out=len(PT),
    n_epochs=2000,
    l_rate=0.05,
    threshold=-1,
)
loss = t.fit(train)


X_test = test[0][0]
ohe_matrix, predicted_tokens = t.predict(X_test)

print(
    "Input: ",
    [EN[idx] for idx, row in enumerate(X_test) for token in row if token == 1.0],
)
print("Predicted:", [PT[token] for token in predicted_tokens])

"""
Input:  ['I', 'want', 'my', 'house', '<EOS>']
Predicted: ['Eu', 'quero', 'minha', 'casa', '<EOS>']

For the following configs
    decoder_n_out=len(PT),
    n_epochs=2000,
    l_rate=0.05,
    threshold=-1,
"""
