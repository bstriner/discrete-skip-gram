import numpy as np


class Prior(object):
    def __init__(self, k, minlen, maxlen):
        self.k = k
        self.minlen = minlen
        self.maxlen = maxlen

    def prior_sample(self):
        length = np.random.randint(self.minlen, self.maxlen + 1, dtype=np.int32)
        symbols = np.random.randint(0, self.k, (length,), dtype=np.int32)
        ret = np.concatenate((symbols + 1, np.zeros((self.maxlen - length,), dtype=np.int32)), axis=0)
        return ret.astype(np.int32)

    def prior_samples(self, n):
        samples = [self.prior_sample() for _ in range(n)]
        return np.vstack(sample.reshape((1, -1)) for sample in samples).astype(np.int32)
