import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from ..utils import build_kernel, build_bias, build_pair


class BiasLayer(Layer):
    def __init__(self,units,
                 bias_initializer='zero', bias_regularizer=None):
        self.units = units
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        Layer.__init__(self)

    def build(self, input_shape):
        self.bias = build_bias(self, (1, self.units), "bias")
        self.built = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def call(self, x):
        n = x.shape[0]
        return T.repeat(self.bias, axis=0, repeats=n)
