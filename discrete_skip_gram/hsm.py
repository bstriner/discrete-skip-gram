import os
import pickle

import numpy as np


class HSM(object):
    def __init__(self, codes, words):
        self.codes = codes
        self.words = words

    def decode(self, code):
        assert len(code.shape) == 1
        d = code.shape[0]
        i = np.sum(np.power(2, np.arange(d)[::-1]) * code)
        if i < self.words.shape[0]:
            return self.words[i]
        else:
            return 0

    def encode(self, id):
        return self.codes[id,:]

    def save(self, path):
        if os.path.exists(path):
            raise ValueError("Path already exists: {}".format(path))
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:
            pickle.dump((self.codes, self.words), f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            codes, words = pickle.load(f)
            return cls(codes=codes, words=words)
