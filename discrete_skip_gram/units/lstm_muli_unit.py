import theano.tensor as T

from ..layers.utils import pair, b, W


class LSTMMultiUnit(object):
    def __init__(self, model, input_units, units, name):
        self.inputs = len(input_units)
        bias = b(model, (units,), "{}_b".format(name))
        Wh = W(model, (units, units), "{}_Wh".format(name))
        Ws = [W(model, (iu, units), "{}_W_{}".format(name, i)) for i, iu in enumerate(input_units)]
        f_W, f_b = pair(model, (units, units), "{}_f".format(name))
        i_W, i_b = pair(model, (units, units), "{}_i".format(name))
        c_W, c_b = pair(model, (units, units), "{}_c".format(name))
        o_W, o_b = pair(model, (units, units), "{}_o".format(name))
        self.non_sequences = [bias, Wh] + Ws + [
            f_W, f_b,
            i_W, i_b,
            c_W, c_b,
            o_W, o_b
        ]
        self.count = len(self.non_sequences)
        self.h0 = b(model, (1, units), "{}_h0".format(name))

    def call(self, h0, xs, params):
        assert(len(xs) == self.inputs)
        idx = 0
        h = params[idx]
        idx += 1
        Wh = params[idx]
        idx += 1
        h += T.dot(h0, Wh)
        for x in xs:
            w = params[idx]
            idx += 1
            h += T.dot(x, w)
        h = T.tanh(h)
        (f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b) = params[idx:(idx+8)]
        idx += 8
        assert idx == len(params)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        o1 = o * h1
        return h1, o1
