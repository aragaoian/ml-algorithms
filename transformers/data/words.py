import numpy as np

EN = ["I", "love", "see", "want", "my", "girlfriend", "dog", "house", "book", "<EOS>"]
PT = [
    "Eu",
    "amo",
    "vejo",
    "quero",
    "meu",
    "minha",
    "namorada",
    "cachorro",
    "casa",
    "livro",
    "<EOS>",
]

EN_IDX = {w: i for i, w in enumerate(EN)}
PT_IDX = {w: i for i, w in enumerate(PT)}

PAIRS = [
    ("I love my girlfriend", "Eu amo minha namorada"),
    ("I love my dog", "Eu amo meu cachorro"),
    ("I love my house", "Eu amo minha casa"),
    ("I love my book", "Eu amo meu livro"),
    ("I see my girlfriend", "Eu vejo minha namorada"),
    ("I see my dog", "Eu vejo meu cachorro"),
    ("I see my house", "Eu vejo minha casa"),
    ("I see my book", "Eu vejo meu livro"),
    ("I want my girlfriend", "Eu quero minha namorada"),
    ("I want my dog", "Eu quero meu cachorro"),
    ("I want my house", "Eu quero minha casa"),
    ("I want my book", "Eu quero meu livro"),
    ("I want my house", "Eu quero minha casa"),
    ("I see my book", "Eu vejo meu livro"),
]

# TODO
# one-hot encode rows and cols
# train_test _split
