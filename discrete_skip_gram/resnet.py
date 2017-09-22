import keras.backend as K
import theano.tensor as T

from .tensor_util import leaky_relu


class Resnet(object):
    def __init__(self,
                 input_units,
                 res_units,
                 depth,
                 initializer,
                 activation=leaky_relu):
        self.activation = activation
        self.depth = depth
        self.input_units=input_units
        self.params = []
        self.ws = []
        for i in range(depth):
            w1 = K.variable(initializer((input_units, res_units)))
            self.params.append(w1)
            self.params.append(K.variable(initializer((res_units,))))
            w2 = K.variable(initializer((res_units, res_units)))
            self.params.append(w2)
            self.params.append(K.variable(initializer((res_units,))))
            w3 = K.variable(initializer((res_units, input_units)))
            self.params.append(w3)
            self.ws.append(w1)
            self.ws.append(w2)
            self.ws.append(w3)

    def call(self, x):
        return self.call_on_params(x, self.params)

    def call_on_params(self, x, params):
        h = x
        n = 5
        assert len(params) == self.depth * n
        for i in range(self.depth):
            r1 = self.activation(T.dot(h, params[(i * n) + 0]) + params[(i * n) + 1])
            r2 = self.activation(T.dot(r1, params[(i * n) + 2]) + params[(i * n) + 3])
            r3 = T.dot(r2, params[(i * n) + 4])
            h += r3
        return h
