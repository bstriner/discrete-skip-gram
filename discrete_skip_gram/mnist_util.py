import os

import numpy as np
from keras.datasets import mnist
from .util import make_path


def binarize(x):
    rnd = np.random.uniform(low=0., high=255., size=x.shape)
    xbin = np.int32(x > rnd)
    return np.reshape(xbin, (-1, 28 * 28))


def binarized():
    (x, y), (xt, yt) = mnist.load_data()
    return binarize(x), binarize(xt)


def mnist_save(path):
    assert not os.path.exists(path)
    make_path(path)
    x, xt = binarized()
    np.savez(path, xtrain=x, xtest=xt)
