from ..layers.utils import build_pair, build_kernel, build_bias, build_beta, build_gamma
from ..layers.utils import build_moving_mean, build_moving_variance
import theano.tensor as T
import itertools
from ..layers.utils import layer_norm

class BatchnormUnit(object):
    def __init__(self,
                 name):
        self.non_sequences = []
        self.count = len(self.non_sequences)

    def call(self, x, params):
        idx = 0
        assert idx == len(params)
        out_test = T.nnet.bn.batch_normalization_test(x, 1., 0., axes=0)
        return out_test
"""
    def call3d(self, x, params):
        assert x.ndim == 3
        (h_W, h_b) = params
        y = T.tensordot(x, h_W, axes=[2, 0]) + (h_b.dimshuffle(('x', 'x', 0)))
        if self.activation:
            y = self.activation(y)
        return y
"""