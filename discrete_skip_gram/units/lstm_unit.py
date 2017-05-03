import theano.tensor as T

from ..layers.utils import pair, b


class LSTMUnit(object):
    def __init__(self, model, units, name):
        h_W, h_b = pair(model, (units, units), "{}_h".format(name))
        f_W, f_b = pair(model, (units, units), "{}_f".format(name))
        i_W, i_b = pair(model, (units, units), "{}_i".format(name))
        c_W, c_b = pair(model, (units, units), "{}_c".format(name))
        o_W, o_b = pair(model, (units, units), "{}_o".format(name))
        self.non_sequences = [
            h_W, h_b,
            f_W, f_b,
            i_W, i_b,
            c_W, c_b,
            o_W, o_b
        ]
        self.count = len(self.non_sequences)
        self.h0 = b(model, (1, units), "{}_h0".format(name))

    def call(self, h0, haddl, params):
        (h_W, h_b,
         f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b) = params
        h = T.tanh(T.dot(h0, h_W)+h_b+haddl)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        o1 = o * h1
        return h1, o1
