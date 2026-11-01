import numpy as np
import numpy.typing as npt

EOS_TAG = "<EOS>"
EN = ["I", "love", "see", "want", "my", "girlfriend", "dog", "house", "book", EOS_TAG]
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
    EOS_TAG,
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


def tokens(sentence: str) -> list[str]:
    return sentence.split() + [EOS_TAG]


def ohe_columns(words: list, indexes: dict) -> npt.NDArray:
    arr = np.zeros((len(indexes), len(words)))
    for pos, word in enumerate(words):
        arr[indexes[word]][pos] = 1.0

    return arr


def encode_pairs(pairs: list[tuple], split: float = 0.2) -> tuple[list[tuple]]:
    # NOTE
    # for the sake of reproducibility and simplicity the split
    # will be done by separating roughly 20% (floored) of the dataset
    # (the last 2 pairs)
    split_idx = int(np.floor(len(pairs) * split))
    encoded_pairs = []

    for first, second in pairs:
        source = tokens(first)
        target = tokens(second)
        decoder_input = [EOS_TAG] + tokens(second)[:-1]

        encoded_source = ohe_columns(source, EN_IDX)
        encoded_target = ohe_columns(target, PT_IDX)
        encoded_decoder_input = ohe_columns(decoder_input, PT_IDX)

        encoded_pairs.append((encoded_source, encoded_decoder_input, encoded_target))

    # NOTE
    # return tuple(train, test)
    return encoded_pairs[:-split_idx], encoded_pairs[-split_idx:]
