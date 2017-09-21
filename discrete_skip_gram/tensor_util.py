import h5py
import theano.tensor as T

import keras.backend as K
from .util import latest_file
from .util import make_path


def softmax_nd(x, axis=-1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / T.sum(e_x, axis=axis, keepdims=True)
    return out

def smoothmax_nd(x, axis=-1, keepdims=True):
    sm = softmax_nd(x, axis=axis)
    return T.sum(sm*x, axis=axis, keepdims=keepdims)

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
    path, epoch = latest_file(dir_path, fmt)
    if path:
        print("Loading epoch {}: {}".format(epoch, path))
        load_weights(path, weights)
        return epoch + 1
    else:
        return 0

def leaky_relu(x):
    return T.nnet.relu(x, 0.2)