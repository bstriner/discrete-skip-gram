import theano.tensor as T
from keras.engine import InputSpec
from keras.layers import Layer
import theano

class SamplerDeterministicLayer(Layer):
    def __init__(self, offset=None):
        self.offset = offset
        self.input_spec = InputSpec(min_ndim=2)
        Layer.__init__(self)

    def build(self, p):
        self.built = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def call(self, p):
        sample = T.argmax(p, axis=1, keepdims=True)
        if self.offset:
            sample += self.offset
        sample = T.cast(sample, dtype='int32')
        sample = theano.gradient.zero_grad(sample)
        return sample
