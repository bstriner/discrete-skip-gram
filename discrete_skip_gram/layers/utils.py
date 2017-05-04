import numpy as np
from keras.layers import Lambda
from theano import tensor as T


def W(model, shape, name):
    return model.add_weight(shape,
                            initializer=model.kernel_initializer,
                            name=name,
                            regularizer=model.kernel_regularizer)


def b(model, shape, name):
    return model.add_weight(shape,
                            initializer=model.bias_initializer,
                            name=name,
                            regularizer=model.bias_regularizer)


def embedding(model, shape, name, dtype='float32'):
    return model.add_weight(shape,
                            initializer=model.embeddings_initializer,
                            name=name,
                            regularizer=model.embeddings_regularizer,
                            dtype=dtype)


def pair(model, shape, name):
    return W(model, shape, "{}_W".format(name)), b(model, (shape[1],), "{}_b".format(name))


def shift_tensor_layer():
    return Lambda(shift_tensor, output_shape=lambda _x: _x)


def shift_tensor(_x):
    return T.concatenate((T.zeros_like(_x[:,0:1,...]), _x[:, :-1,...] + 1), axis=1)


def softmax_nd(x, axis=-1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out


def softmax_nd_layer():
    return Lambda(softmax_nd, output_shape=lambda _x: _x)


def custom_loss(_, y_pred):
    return T.mean(y_pred, axis=None)


def rewards_to_values(r, discount=0.75):
    cum = np.zeros((r.shape[0],), dtype=np.float32)
    ret = np.zeros(r.shape, dtype=np.float32)
    for i in range(r.shape[1]):
        v = cum * discount + r[:, r.shape[1] - 1 - i]
        ret[:, r.shape[1] - 1 - i] = v
        cum = v
    return ret


def drop_dim_2():
    return Lambda(lambda _x: _x[:, 0, :], output_shape=lambda _x: (_x[0], _x[2]))


def zeros_layer(units, dtype='float32'):
    return Lambda(lambda _x: T.zeros((_x.shape[0], units), dtype=dtype), output_shape=lambda _x: (_x[0], units))


def ones_layer(units, scale=1, dtype='float32'):
    return Lambda(lambda _x: T.ones((_x.shape[0], units), dtype=dtype) * scale, output_shape=lambda _x: (_x[0], units))


def add_layer(value):
    return Lambda(lambda _x: _x + value, output_shape=lambda _x: _x)
