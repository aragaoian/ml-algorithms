import numpy as np

from transformers.positional_encoder import PositionalEncoder
from transformers.word_embedding import WordEmbedding

VOCAB = ["I", "love", "my", "girlfriend", "<EOS>"]

eye = np.eye(5)
X = eye[:, :4]  # I, love, my, girlfriend
y = eye[:, 1:]  # love, my, girlfriend, <EOS>

we = WordEmbedding(X, y)
we.fit()
embeddings = we.embeddings()

pe = PositionalEncoder(embeddings)
positional_encoded_embeddings = pe.generate()
