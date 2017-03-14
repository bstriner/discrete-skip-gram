import itertools
import numpy as np
from .corpora import clean


class Dataset(object):
    def __init__(self, words):
        self.words = list(words)
        self.unique_words = len(set(self.words))
        self.depth = max(len(word) for word in words)
        self.charset = list(sorted(list(set(itertools.chain.from_iterable(self.words)))))
        self.charmap = {char: i for i, char in enumerate(self.charset)}
        self.x_k = len(self.charset)
        self.word_matrix = self.words_to_matrix(self.words)
        print("Dataset words: {}, unique words: {}, characters: {}, max length: {}".format(
            len(self.words), self.unique_words, self.x_k, self.depth))

    def word_to_vector(self, word):
        assert len(word) <= self.depth
        ar = [self.charmap[c] + 1 for c in word]
        while len(ar) < self.depth:
            ar.append(0)
        vec = np.array(ar).astype(np.int32)
        return vec

    def vector_to_word(self, vec):
        assert vec.ndim == 1
        str = ""
        for i in range(vec.shape[0]):
            c = vec[i]
            if c > 0:
                str += self.charset[c - 1]
        return str

    def matrix_to_words(self, mat):
        n = mat.shape[0]
        return [self.vector_to_word(mat[i, :]) for i in range(n)]

    def words_to_matrix(self, words):
        return np.vstack([self.word_to_vector(word).reshape((1, -1)) for word in words]).astype(np.int32)

    def sample_skip_grams(self, n, window):
        data = [self.sample_skip_gram(window) for _ in range(n)]
        x = np.vstack(d[0].reshape((1, -1)) for d in data)
        xnoised = np.vstack(d[1].reshape((1, -1)) for d in data)
        return x, xnoised


class DatasetSingle(Dataset):
    def __init__(self, words):
        self.words = list(words)
        self.unique_words = len(set(self.words))
        self.depth = max(len(word) for word in words)
        self.charset = list(sorted(list(set(itertools.chain.from_iterable(self.words)))))
        self.charmap = {char: i for i, char in enumerate(self.charset)}
        self.x_k = len(self.charset)
        self.word_matrix = self.words_to_matrix(self.words)

    def sample_skip_gram(self, window):
        ind = np.random.randint(0, self.word_matrix.shape[0])
        offset = np.random.randint(-window, window + 1)
        ind2 = np.clip(ind + offset, 0, self.word_matrix.shape[0] - 1)
        return self.word_matrix[ind, :], self.word_matrix[ind2, :]

    def sample_vectors(self, n):
        idx = np.random.randint(0, self.word_matrix.shape[0], (n,))
        return self.word_matrix[idx, :]

    def summary(self):
        print("Dataset words: {}, unique words: {}, characters: {}, max length: {}".format(
            len(self.words), self.unique_words, self.x_k, self.depth))


class DatasetFiles(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus
        self.words = list(itertools.chain.from_iterable(clean(corpus.words(f)) for f in corpus.fileids()))
        self.unique_words = len(set(self.words))
        self.depth = max(len(word) for word in self.words)
        self.charset = list(sorted(list(set(itertools.chain.from_iterable(self.words)))))
        self.charmap = {char: i for i, char in enumerate(self.charset)}
        self.x_k = len(self.charset)
        self.fileids = list(corpus.fileids())
        self.filecount = len(self.fileids)
        self.files = {fileid: self.words_to_matrix(clean(self.corpus.words(fileid))) for fileid in self.fileids}

    def summary(self):
        print("Dataset words: {}, unique words: {}, characters: {}, max length: {}, file count: {}".format(
            len(self.words), self.unique_words, self.x_k, self.depth, len(self.fileids)))

    def sample_fileid(self):
        ind = np.random.randint(0, self.filecount)
        return self.fileids[ind]

    def sample_skip_gram(self, window):
        fileid = self.sample_fileid()
        words = self.files[fileid]
        wordcount = words.shape[0]
        ind = np.random.randint(0, wordcount)
        w1 = words[ind, :]
        if window == 0:
            return w1, w1
        else:
            indmin = max(0, ind - window)
            indmax = min(wordcount - 1, ind + window)
            opts = list(range(indmin, ind)) + list(range(ind + 1, indmax + 1))
            optlen = len(opts)
            assert optlen > 0
            x = np.random.randint(0, optlen)
            ind2 = opts[x]
            w2 = words[ind2, :]
            return w1, w2

    def sample_vector(self):
        fileid = self.sample_fileid()
        words = self.files[fileid]
        k = words.shape[0]
        idx = np.random.randint(0, k)
        return words[idx:idx + 1, :]

    def sample_vectors(self, n):
        return np.vstack(list(self.sample_vector() for _ in range(n)))
