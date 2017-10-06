import numpy as np

from .util import one_hot_np


def calc_initial_pz(encoding, z_k, smoothing=0.1, noise=0):
    w = one_hot_np(encoding, k=z_k)
    w = (w * (1. - smoothing)) + (smoothing / z_k)
    w = np.log(w)
    w -= np.max(w, axis=1, keepdims=True)
    if noise > 0:
        w += np.random.uniform(low=-noise, high=noise, size=w.shape)
    return w


def calc_initial_pz_random(x_k, z_k, smoothing=0.1, noise=0):
    encoding = np.random.random_integers(low=0, high=z_k - 1, size=(x_k,))
    return calc_initial_pz(encoding=encoding,
                           z_k=z_k,
                           smoothing=smoothing,
                           noise=noise)
