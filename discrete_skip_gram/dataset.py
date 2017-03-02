import itertools
import numpy as np


class Dataset(object):
    def __init__(self, words):
        self.words = list(words)
        self.depth = max(len(word) for word in words)
        self.charset = list(sorted(list(set(itertools.chain.from_iterable(self.words)))))
        self.charmap = {char: i for i, char in enumerate(self.charset)}
        self.x_k = len(self.charset)
        self.word_matrix = self.words_to_matrix(self.words)
        print("Dataset words: {}, characters: {}, max length: {}".format(len(self.words), self.x_k, self.depth))

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

    def sample_skip_gram(self, window):
        ind = np.random.randint(0, self.word_matrix.shape[0])
        offset = np.random.randint(-window, window + 1)
        ind2 = np.clip(ind + offset, 0, self.word_matrix.shape[0] - 1)
        return self.word_matrix[ind, :], self.word_matrix[ind2, :]

    def sample_skip_grams(self, n, window):
        data = [self.sample_skip_gram(window) for _ in range(n)]
        x = np.vstack(d[0].reshape((1, -1)) for d in data)
        xnoised = np.vstack(d[1].reshape((1, -1)) for d in data)
        return x, xnoised

    def sample_vectors(self, n):
        idx = np.random.randint(0, self.word_matrix.shape[0], (n,))
        return self.word_matrix[idx, :]
