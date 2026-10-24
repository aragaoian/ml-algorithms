import numpy as np

from transformers.transformer import Transformer

VOCAB = ["I", "love", "my", "girlfriend", "<EOS>"]

eye = np.eye(5)
X = eye[:, :4]  # I, love, my, girlfriend
y = eye[:, 1:]  # love, my, girlfriend, <EOS>

t = Transformer(len(VOCAB))
t.fit(X, y)
