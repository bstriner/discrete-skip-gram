import theano.tensor as T
from keras.engine import InputSpec
from keras.layers import Layer
import theano


class SamplerLayer(Layer):
    def __init__(self, srng, offset=None):
        self.srng = srng
        self.offset = offset
        self.input_spec = InputSpec(min_ndim=2)
        Layer.__init__(self)

    def build(self, p):
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return (input_shape[0], 1)

    def call(self, p):
        csum = T.cumsum(p, axis=1)
        n = p.shape[0]
        rng = self.srng.uniform(low=0, high=1, size=(n,), dtype='float32')
        # print "CSuM ndim: {}".format(csum.ndim)
        sample = T.sum(T.gt(rng.dimshuffle((0, 'x')), csum), axis=1, keepdims=True)
        if self.offset:
            sample += self.offset
        sample = T.cast(sample, dtype='int32')
        sample = theano.gradient.zero_grad(sample)
        return sample
