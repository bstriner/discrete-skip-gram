import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import build_kernel, build_bias, build_pair, shift_tensor


class ShiftPaddingLayer(Layer):
    """
    Shift dim 2 and pad with learned bias
    """

    def __init__(self,
                 bias_initializer='random_uniform', bias_regularizer=None):
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = [InputSpec(ndim=3)]
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[2]
        self.h0 = build_bias(self, (1, 1, input_dim), "h0")
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        return input_shape

    def call(self, x):
        n = x.shape[0]
        x0 = T.extra_ops.repeat(self.h0, n, axis=0)
        x1 = x[:, :-1, :]
        y = T.concatenate((x0, x1), axis=1)
        return y
