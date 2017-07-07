import theano.tensor as T


def softmax_nd(x, axis=-1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    #    eps = 1e-7
    #    out = e_x / (e_x.sum(axis=axis, keepdims=True) + eps)
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out
