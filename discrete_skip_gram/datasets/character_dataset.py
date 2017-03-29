import itertools
import numpy as np


# Words: 1161192, Unique: 34544
# Words: 1161192, Unique: 49815


def get_charset(words):
    charset = list(set(itertools.chain.from_iterable(words)))
    charset.sort()
    charmap = {c: i for i, c in enumerate(charset)}
    return charset, charmap


def word_to_array(word, charmap):
    x = np.zeros((len(word) + 1,))
    for i, c in enumerate(word):
        x[i] = charmap[c] + 1
    return x


def docs_to_arrays(docs, charmap):
    return [[word_to_array(word, charmap) for word in doc] for doc in docs]


def sample_skip_gram(adocs, window):
    doc_idx = np.random.randint(0, len(adocs))
    doc = adocs[doc_idx]
    word_idx = np.random.randint(0, len(doc))
    word = doc[word_idx]
    if window == 0 or len(doc) == 1:
        return word, word
    else:
        assert len(doc) > 1
        min_idx = max(0, word_idx - window)
        max_idx = min(len(doc) - 1, word_idx + window)
        choices = list(range(min_idx, word_idx) + range(word_idx + 1, max_idx + 1))
        choice_idx = np.random.randint(0, len(choices))
        choice = choices[choice_idx]
        word_context = doc[choice]
        return word, word_context


def skip_gram_batch(adocs, window, n):
    grams = [sample_skip_gram(adocs, window) for _ in range(n)]
    depth_x = max(len(g[0]) for g in grams)
    depth_y = max(len(g[1]) for g in grams)
    xs = np.zeros((n, depth_x), dtype=np.int32)
    ys = np.zeros((n, depth_y), dtype=np.int32)
    for i, (x, y) in enumerate(grams):
        xs[i, :x.shape[0]] = x + 1
        ys[i, :y.shape[0]] = y + 1
    return [xs, ys], [np.zeros((n, 1), dtype=np.float32)]


def skip_gram_generator(adocs, window, n):
    while True:
        yield skip_gram_batch(adocs, window, n)
