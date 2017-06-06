import numpy as np
from keras.layers import Lambda
from theano import tensor as T


def build_kernel(model, shape, name):
    return model.add_weight(shape,
                            initializer=model.kernel_initializer,
                            name=name,
                            regularizer=model.kernel_regularizer)


def build_bias(model, shape, name):
    return model.add_weight(shape,
                            initializer=model.bias_initializer,
                            name=name,
                            regularizer=model.bias_regularizer)


def build_beta(model, shape, name):
    return model.add_weight(shape,
                            initializer=model.beta_initializer,
                            name=name)


def build_gamma(model, shape, name):
    return model.add_weight(shape,
                            initializer=model.gamma_initializer,
                            name=name)


def build_moving_mean(model, shape, name):
    return model.add_weight(shape,
                            initializer=model.moving_mean_initializer,
                            name=name)


def build_moving_variance(model, shape, name):
    return model.add_weight(shape,
                            initializer=model.moving_variance_initializer,
                            name=name)


def build_embedding(model, shape, name, dtype='float32'):
    return model.add_weight(shape,
                            initializer=model.embeddings_initializer,
                            name=name,
                            regularizer=model.embeddings_regularizer,
                            dtype=dtype)


def build_pair(model, shape, name):
    return [build_kernel(model, shape, "{}_W".format(name)), build_bias(model, (shape[1],), "{}_b".format(name))]


def shift_tensor_layer():
    return Lambda(shift_tensor, output_shape=lambda _x: _x)


def shift_tensor(_x):
    return T.concatenate((T.zeros_like(_x[:, 0:1, ...]), _x[:, :-1, ...] + 1), axis=1)


def softmax_nd(x, axis=-1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    #    eps = 1e-7
    #    out = e_x / (e_x.sum(axis=axis, keepdims=True) + eps)
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out


def softmax_nd_layer(**kwargs):
    return Lambda(softmax_nd, output_shape=lambda _x: _x, **kwargs)


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


def layer_norm(x):
    mean = T.mean(x, axis=-1, keepdims=True)
    std = T.std(x, axis=-1, keepdims=True)
    eps = 1e-6
    return (x - mean) / (std + eps)


def leaky_relu(x):
    return T.nnet.relu(x, 0.2)


def nll_metric(avg_nll, idx):
    def metric(ytrue, ypred):
        return avg_nll[idx]

    metric.__name__ = "nll{:02d}".format(idx)
    return metric


def nll_metrics(avg_nll, z_depth):
    return [nll_metric(avg_nll, i) for i in range(z_depth)]


def sample_3d(x):
    return T.argmax(x, axis=2)


def sample_3d_layer(**kwargs):
    return Lambda(lambda _x: sample_3d(_x), output_shape=lambda _x: (_x[0], _x[1]), **kwargs)


def uniform_smoothing(x, factor=1e-9):
    factor = np.float32(factor)
    scale = 1.0 - (x.shape[-1] * factor)
    return factor + (scale * x)
