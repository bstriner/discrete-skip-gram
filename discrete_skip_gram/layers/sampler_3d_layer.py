import theano.tensor as T
from keras.engine import InputSpec
from keras.layers import Layer
import theano

class Sampler3DLayer(Layer):
    def __init__(self, srng, offset=None):
        self.srng = srng
        self.offset = offset
        self.input_spec = InputSpec(ndim=3)
        Layer.__init__(self)

    def build(self, p):
        assert len(p)==3
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape)==3
        return input_shape[0], input_shape[1]

    def call(self, p):
        csum = T.cumsum(p, axis=-1)
        shape = p.shape[:-1]
        rng = self.srng.uniform(low=0, high=1, size=shape, dtype='float32')
        sample = T.sum(T.gt(rng.dimshuffle((0, 1, 'x')), csum), axis=-1)
        if self.offset:
            sample += self.offset
        sample = T.cast(sample, dtype='int32')
        sample = theano.gradient.zero_grad(sample)
        return sample