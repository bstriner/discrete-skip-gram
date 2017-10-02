import keras.backend as K
import theano
import theano.tensor as T
from ..tensor_util import softmax_nd

class IndependentParameterization(object):
    def __init__(self,
                 x_k,
                 z_k,
                 initializer,
                 srng):
        self.z_k = z_k

        pz_weight = K.variable(initializer(x_k, z_k))
        pz = softmax_nd(pz_weight)

        cs = T.cumsum(pz, axis=1)
        rnd = srng.uniform(low=0., high=1., dtype='float32', size=(x_k,))
        sel = T.gt(rnd.dimshuffle((0,'x')), cs)
        sel = T.clip(sel, 0, z_k-1)

        p = pz[T.arange(x_k), sel]
        lp = T.sum(T.log(p))
        self.outputs = sel, lp


