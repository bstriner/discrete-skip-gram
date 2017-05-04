import numpy as np


def hsm_decode(code, words):
    assert len(code.shape)==1
    d = code.shape[0]
    i = np.sum(np.power(2, np.arange(d)[::-1]) * code)
    if i < words.shape[0]:
        return words[i]
    else:
        return 0
