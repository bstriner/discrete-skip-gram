import numpy as np

import theano
import theano.tensor as T


def validate_encoding_flat_t(cooccurrence, eps=1e-9):
    _co = cooccurrence.astype(np.float32)
    _co = _co / np.sum(_co, axis=None)
    enc = T.ivector(name="enc")
    z_k = T.iscalar(name="z_k")
    co = T.constant(_co, name="cooccurrence")
    x_k = _co.shape[0]
    m = T.zeros((z_k, x_k))
    m = T.set_subtensor(m[enc, T.arange(x_k)], 1)
    p = T.dot(m, co)  # (z_k, x_k) * (x_k, x_k) = z_k, x_k
    marg = T.sum(p, axis=1, keepdims=True)
    cond = p / (marg + eps)
    nll = T.sum(p * -np.log(eps + cond), axis=None)  # scalar
    f = theano.function(inputs=[enc, z_k], outputs=nll)
    return f
