import numpy as np
from ..datasets.utils import count_words


def doc_to_array(doc, wordmap):
    return np.array([wordmap[w] + 1 if w in wordmap else 0 for w in doc])


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
        if not isinstance(id, int):
            print "Id: {}, {}".format(id, type(id))
        assert isinstance(id, int)
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
