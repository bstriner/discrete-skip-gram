import keras.backend as K
import numpy as np
import theano.tensor as T
from theano.tensor.nnet.bn import batch_normalization_train

from .initializers import uniform_initializer

"""
def bn(x):
    m = T.mean(x, axis=0, keepdims=True)
    s = T.std(x, axis=0, keepdims=True)
    eps = 1e-9
    ret = (x - m) / (s + eps)
    return ret
"""


class MLP(object):
    def __init__(self,
                 input_units,
                 hidden_units,
                 output_units,
                 hidden_depth,
                 initializer=uniform_initializer(0.05),
                 hidden_activation=None,
                 output_activation=None,
                 use_bn=False):
        self.use_bn = use_bn
        self.hidden_depth = hidden_depth
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.params = []

        ws = []
        bs = []

        ws.append(K.variable(initializer((input_units, hidden_units))))
        bs.append(K.variable(initializer((hidden_units,))))
        for i in range(hidden_depth):
            ws.append(K.variable(initializer((hidden_units, hidden_units))))
            bs.append(K.variable(initializer((hidden_units,))))
        ws.append(K.variable(initializer((hidden_units, output_units))))
        bs.append(K.variable(initializer((output_units,))))

        gammas = []
        betas = []
        if self.use_bn:
            for i in range(hidden_depth + 1):
                gammas.append(K.variable(0.1 * np.ones((hidden_units,), dtype=np.float32), dtype='float32'))
                betas.append(K.variable(np.zeros((hidden_units,), dtype=np.float32), dtype='float32'))
        self.ws = ws
        self.bs = bs
        self.params = ws + bs + gammas + betas
        self.gammas = gammas
        self.betas = betas

    def call(self, x):
        return self.call_on_params(x, self.params)

    def call_on_params(self, x, params):
        k = self.hidden_depth + 2
        ws = params[0:k]
        bs = params[k:2 * k]
        if self.use_bn:
            bn = params[2 * k:]
            gammas = bn[:k - 1]
            betas = bn[k - 1:]
        h = T.dot(x, ws[0]) + bs[0]
        if self.hidden_activation:
            h = self.hidden_activation(h)
        if self.use_bn:
            h, _m, _s = batch_normalization_train(h, gamma=gammas[0], beta=betas[0])
        for j in range(self.hidden_depth):
            h = T.dot(h, ws[j + 1]) + bs[j + 1]
            if self.hidden_activation:
                h = self.hidden_activation(h)
            if self.use_bn:
                h, _m, _s = batch_normalization_train(h, gamma=gammas[1 + j], beta=betas[1 + j])
        y = T.dot(h, ws[-1]) + bs[-1]
        if self.output_activation:
            y = self.output_activation(y)
        return y
