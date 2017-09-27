from os.path import exists

import h5py
import keras.backend as K
import theano.tensor as T

from .util import latest_file
from .util import make_path


def softmax_nd(x, axis=-1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / T.sum(e_x, axis=axis, keepdims=True)
    return out


def smoothmax_nd(x, axis=-1, keepdims=True):
    sm = softmax_nd(x, axis=axis)
    return T.sum(sm * x, axis=axis, keepdims=keepdims)


def weight_name(i, weight):
    return "param_{}".format(i)


def save_weights(path, weights):
    make_path(path)
    with h5py.File(path, 'w') as f:
        for i, w in enumerate(weights):
            f.create_dataset(name=weight_name(i, w), data=K.get_value(w))


def load_weights(path, weights):
    with h5py.File(path, 'r') as f:
        for i, w in enumerate(weights):
            K.set_value(w, f[weight_name(i, w)])


def load_latest_weights(dir_path, fmt, weights):
    if exists(dir_path):
        path, epoch = latest_file(dir_path, fmt)
        if path:
            print("Loading epoch {}: {}".format(epoch, path))
            load_weights(path, weights)
            return epoch + 1
    return 0


def leaky_relu(x):
    return T.nnet.relu(x, 0.2)


def tensor_one_hot(x, k):
    assert x.ndim == 1
    assert x.dtype == 'int32' or x.dtype=='int64'
    ret = T.zeros((x.shape[0], k), dtype='float32')
    idx = T.arange(x.shape[0], dtype='int32')
    ret = T.set_subtensor(ret[idx, x], 1.)
    return ret
