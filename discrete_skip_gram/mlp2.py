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


class MLP2(object):
    def __init__(self,
                 input_units,
                 units,
                 initializer=uniform_initializer(0.05),
                 hidden_activation=None,
                 output_activation=None,
                 use_bn=False):
        self.depth = len(units)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.params = []
        self.use_bn = use_bn

        ws = []
        bs = []

        u0 = input_units
        for i in range(self.depth):
            u1 = units[i]
            ws.append(K.variable(initializer((u0, u1))))
            bs.append(K.variable(initializer((u1,))))
            u0 = u1

        """
        gammas = []
        betas = []
        if self.use_bn:
            for u in units[:-1]:
                gammas.append(K.variable(0.1 * np.ones((u,), dtype=np.float32), dtype='float32'))
                betas.append(K.variable(np.zeros((u,), dtype=np.float32), dtype='float32'))
        """
        self.ws = ws
        self.bs = bs
        self.params = ws + bs #+ gammas + betas
        #self.gammas = gammas
        #self.betas = betas

    def call(self, x):
        return self.call_on_params(x, self.params)

    def call_on_params(self, x, params):
        ws = params[:self.depth]
        bs = params[self.depth:]
        h = x
        for i in range(self.depth):
            h = T.dot(h, ws[i])+bs[i]
            if self.hidden_activation and i < self.depth-1:
                h = self.hidden_activation(h)
        if self.output_activation:
            h = self.output_activation(h)
        return h
