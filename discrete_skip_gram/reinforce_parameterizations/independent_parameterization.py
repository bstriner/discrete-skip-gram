import keras.backend as K
import theano.tensor as T

from ..tensor_util import softmax_nd


class IndependentParameterization(object):
    def __init__(self,
                 x_k,
                 z_k,
                 initializer,
                 srng,
                 pz_weight_regularizer=None,
                 pz_regularizer=None):
        self.z_k = z_k

        pz_weight = K.variable(initializer((x_k, z_k)))
        pz = softmax_nd(pz_weight)

        cs = T.cumsum(pz, axis=1)
        rnd = srng.uniform(low=0., high=1., dtype='float32', size=(x_k,))
        sel = T.sum(T.gt(rnd.dimshuffle((0, 'x')), cs), axis=1)
        sel = T.clip(sel, 0, z_k - 1)

        p = pz[T.arange(x_k), sel]
        lp = T.sum(T.log(p))
        self.encoding = sel
        self.logpz = lp
        assert self.encoding.ndim == 1
        assert self.logpz.ndim == 0
        self.params = [pz_weight]
        self.weights = [pz_weight]
        self.loss = T.constant(0.)
        self.regularize = False
        if pz_weight_regularizer:
            self.loss += pz_weight_regularizer(pz_weight)
            self.regularize = True
        if pz_regularizer:
            self.loss += pz_regularizer(pz)
            self.regularize = True
