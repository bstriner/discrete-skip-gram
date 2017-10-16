import itertools

import keras.backend as K
import theano.tensor as T


class LSTMUnit(object):
    def __init__(self,
                 input_units,
                 units,
                 initializer):
        self.input_units = input_units
        self.units = units
        self.initializer = initializer
        self.h0 = K.variable(initializer((1, units)))
        self.recurrent_params = list(itertools.chain.from_iterable([self.variable_set() for _ in range(4)]))
        self.params = [self.h0] + self.recurrent_params

    def variable_set(self):
        wh = K.variable(self.initializer((self.units, self.units)))
        b = K.variable(self.initializer((self.units,)))
        ws = [K.variable(self.initializer((u, self.units))) for u in self.input_units]
        return [wh, b] + ws

    def calc(self, xs, h0, params):
        ret = T.dot(h0, params[0]) + params[1]
        for i, x in enumerate(xs):
            ret += T.dot(x, params[i + 2])
        return ret

    def step(self, xs, h0, params):
        assert len(xs) == len(self.input_units)
        k = len(self.input_units) + 2
        assert len(params) == k * 4
        funs = [T.nnet.sigmoid, T.nnet.sigmoid, T.tanh, T.nnet.sigmoid]
        sets = [params[k * i:k * (i + 1)] for i in range(4)]
        f, i, c, o = [fun(self.calc(xs, h0, p)) for fun, p in zip(funs, sets)]
        h1 = (f * h0) + (i * c)
        y1 = o * T.tanh(h1)
        return [h1, y1]
