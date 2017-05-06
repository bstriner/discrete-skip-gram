from ..layers.utils import pair
import theano.tensor as T

class DenseUnit(object):
    def __init__(self, model, input_dim, units, name, activation=None):
        self.activation = activation
        h_W, h_b = pair(model, (input_dim, units), name)
        self.non_sequences = [h_W, h_b]

    def call(self, x, params):
        assert x.ndim == 2
        (h_W, h_b) = params
        y = T.dot(x, h_W)+h_b
        if self.activation:
            y = self.activation(y)
        return y

    def call3d(self, x, params):
        assert x.ndim == 3
        (h_W, h_b) = params
        y = T.tensordot(x, h_W, axes=[2, 0]) + (h_b.dimshuffle(('x','x',0)))
        if self.activation:
            y = self.activation(y)
        return y
