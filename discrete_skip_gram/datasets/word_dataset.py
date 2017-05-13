import numpy as np
from ..datasets.utils import count_words


def doc_to_array(doc, wordmap):
    return np.array([wordmap[w] + 1 if w in wordmap else 0 for w in doc])


def docs_to_arrays(docs, wordmap):
    return [doc_to_array(doc, wordmap) for doc in docs]


def sample_skip_gram(adocs, window):
    doc_idx = np.random.randint(0, len(adocs))
    doc = adocs[doc_idx]
    assert doc.shape[0] > window
    word_idx = np.random.randint(0+window, doc.shape[0]-window)
    word = doc[word_idx]
    if window == 0 or doc.shape[0] == 1:
        return word, word
    else:
        direction = (np.random.randint(0, 2)*2) - 1
        offset = np.random.randint(1, window+1)
        ctx_idx = word_idx + (offset*direction)
        ctx = doc[ctx_idx]
        return word, ctx


def skip_gram_batch(adocs, window, n):
    grams = [sample_skip_gram(adocs, window) for _ in range(n)]
    xs = np.zeros((n, 1), dtype=np.int32)
    ys = np.zeros((n, 1), dtype=np.int32)
    for i, (x, y) in enumerate(grams):
        xs[i,0] = x
        ys[i,0] = y
    return [xs, ys]


def skip_gram_generator(adocs, window, n):
    while True:
        yield skip_gram_batch(adocs, window, n)


def skip_gram_ones_generator(adocs, window, n):
    while True:
        yield [list(skip_gram_batch(adocs, window, n)), np.ones((n, 1), dtype=np.float32)]

def sample_cbow(docs, window):
    docidx = np.random.randint(0, len(docs))
    doc = docs[docidx]
    minidx = window
    maxidx = doc.shape[0] - window - 1
    idx = np.random.randint(minidx, maxidx + 1)
    word = doc[idx]
    ctx = np.concatenate((doc[(idx - window):idx], doc[(idx + 1):(idx + window + 1)]), axis=0)
    return ctx.reshape((1, -1)), word.reshape((1, -1))

def cbow_batch(docs, n, window):
    gs = [sample_cbow(docs, window) for _ in range(n)]
    x = np.vstack(g[0] for g in gs)
    y = np.vstack(g[1] for g in gs)
    return [x, y]



class WordDataset(object):
    def __init__(self, docs, min_count, tdocs=None):
        self.count = sum(len(doc) for doc in docs)
        self.wordcounts = count_words(docs)
        self.min_count = min_count
        self.wordset = [k for k, v in self.wordcounts.iteritems() if v >= self.min_count]
        self.wordset.sort()
        self.wordmap = {w: i for i, w in enumerate(self.wordset)}
        self.adocs = docs_to_arrays(docs, self.wordmap)
        self.k = len(self.wordset) + 1
        self.tdocs=tdocs
        if tdocs:
            self.tadocs = docs_to_arrays(tdocs, self.wordmap)


    def summary(self):
        print "Total wordcount: {}".format(self.count)
        print "Unique words: {}, Filtered: {}".format(len(self.wordcounts), len(self.wordset))

    def get_word(self, id):
        if id >= self.k:
            raise ValueError("Invalid word: {}>={}".format(id, self.k))
        return "_UNK_" if id == 0 else self.wordset[id - 1]

    def skip_gram_batch(self, n, window, test=False):
        return skip_gram_batch(self.tadocs if test else self.adocs, window=window, n=n)

    def unigram_generator(self, n):
        while True:
            sg = self.skip_gram_batch(window=0, n=n)
            yield sg[0], np.ones((n, 1), dtype=np.float32)

    def bigram_generator(self, n, window):
        while True:
            sg = self.skip_gram_batch(window=window, n=n)
            yield sg, np.ones((n, 1), dtype=np.float32)

    def skipgram_generator(self, n, window):
        while True:
            sg = self.skip_gram_batch(window=window, n=n)
            yield sg, np.ones((n, 1), dtype=np.float32)

    def cbow_batch(self, n, window, test=False):
        return cbow_batch(self.tadocs if test else self.adocs, n, window)

    def cbow_generator(self, n, window):
        while True:
            x, y = self.cbow_batch(n, window)
            yield [x, y], np.ones((n, 1), dtype='float32')

    def skipgram_generator_with_context(self, n, window):
        while True:
            x, y = self.cbow_batch(n, window)
            yield [y, x], np.ones((n, 1), dtype='float32')

    def skipgram_generator_no_context(self, n, window):
        while True:
            x, y = self.cbow_batch(n, window)
            yield x, np.ones((n, 1), dtype='float32')
