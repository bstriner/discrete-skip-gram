import numpy as np

def doc_to_array(doc, wordmap):
    return np.array([wordmap[w]+1 if w in wordmap else 0 for w in doc])

def docs_to_arrays(docs, wordmap):
    return [doc_to_array(doc, wordmap) for doc in docs]

def sample_skip_gram(adocs, window):
    doc_idx = np.random.randint(0, len(adocs))
    doc = adocs[doc_idx]
    word_idx = np.random.randint(0, doc.shape[0])
    word = doc[word_idx]
    if window == 0 or doc.shape[0] == 1:
        return word, word
    else:
        assert doc.shape[0] > 1
        min_idx = max(0, word_idx - window)
        max_idx = min(doc.shape[0] - 1, word_idx + window)
        choices = list(range(min_idx, word_idx) + range(word_idx + 1, max_idx + 1))
        choice_idx = np.random.randint(0, len(choices))
        choice = choices[choice_idx]
        word_context = doc[choice]
        return word, word_context


def skip_gram_batch(adocs, window, n):
    grams = [sample_skip_gram(adocs, window) for _ in range(n)]
    xs = np.zeros((n, 1), dtype=np.int32)
    ys = np.zeros((n, 1), dtype=np.int32)
    for i, (x, y) in enumerate(grams):
        xs[i] = x
        ys[i] = y
    return xs, ys


def skip_gram_generator(adocs, window, n):
    while True:
        yield skip_gram_batch(adocs, window, n)

def skip_gram_ones_generator(adocs, window, n):
    while True:
        yield [list(skip_gram_batch(adocs, window, n)), np.ones((n,1), dtype=np.float32)]

