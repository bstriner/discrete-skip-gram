import theano
import theano.tensor as T
from keras import activations
from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer

from .utils import pair, W, b


class DenseBatch(Layer):
    """
    
    """

    def __init__(self, units,
                 activation=None,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None,
                 **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = False
        Layer.__init__(self, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        depth = input_shape[1]
        input_dim = input_shape[2]
        self.kernel = W(self, (depth, input_dim, self.units), "kernel")
        self.bias = b(self, (depth, self.units), "bias")
        self.built = True

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        return (input_shape[0], input_shape[1], self.units)

    def call(self, x):
        # x: (n, depth, input_dim)
        # kernel: (depth, input_dim, units)
        xr = T.transpose(x,(1,0,2)) # depth, n, input_dim
        h = T.batched_tensordot(xr, self.kernel, axes=[2, 1]) # depth, n, units
        h = T.transpose(h, (1,0,2)) # n, depth, units
        y = h + self.bias
        if self.activation:
            y = self.activation(y)
        return y
