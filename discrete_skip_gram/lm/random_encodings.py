import itertools

import numpy as np


def random_encodings(x_k, p, units):
    if 2 ** units < x_k:
        raise ValueError("x_k ({}) must be <= 2^units ({})".format(x_k, 2 ** units))

    codes = []
    for i in range(x_k):
        c = list(np.random.binomial(n=1, p=p, size=(units,)))
        while c in codes:
            c = list(np.random.binomial(n=1, p=p, size=(units,)))
        codes.append(c)
    codes = np.array(codes, dtype=np.int32)
    np.random.shuffle(codes)
    return codes


def random_encodings_uniform(x_k, units):
    # only for small number of units
    if 2 ** units < x_k:
        raise ValueError("x_k ({}) must be <= 2^units ({})".format(x_k, 2 ** units))

    codes = list(list(i) for i in itertools.product(range(2), repeat=units))
    codes = np.array(codes)
    np.random.shuffle(codes)
    codes = codes[:x_k, :]
    return codes
