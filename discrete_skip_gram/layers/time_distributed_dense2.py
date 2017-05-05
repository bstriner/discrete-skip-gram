import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import W, b, pair
from keras import activations

class TimeDistributedDense2(Layer):
    """
    
    """

    def __init__(self, units,
                 activation = None,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[2]

        self.kernel, self.bias = pair(self, (input_dim, self.units), "h")
        self.built = True

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        return (input_shape[0], input_shape[1], self.units)

    def call(self, x):
        y = T.tensordot(x, self.kernel, axes=[2, 0])+(self.bias.dimshuffle(('x','x',0)))
        if self.activation:
            y = self.activation(y)
        return y
