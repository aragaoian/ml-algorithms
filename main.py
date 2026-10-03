import numpy as np

from transformers.word_embedding import WordEmbedding

VOCAB = ["I", "love", "my", "girlfriend", "<EOS>"]

eye = np.eye(5)
X = eye[:, :4]  # I, love, my, girlfriend
y = eye[:, 1:]  # love, my, girlfriend, <EOS>

we = WordEmbedding(X, y)
we.fit()
