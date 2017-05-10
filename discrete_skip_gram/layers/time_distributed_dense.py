import theano
import theano.tensor as T
from keras import activations
from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer

from .utils import pair


class TimeDistributedDense(Layer):
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
        input_dim = input_shape[2]

        h_W, h_b = pair(self, (input_dim, self.units), "h")
        self.kernel = h_W
        self.bias = h_b
        self.non_sequences = [
            h_W, h_b
        ]
        self.built = True

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        return (input_shape[0], input_shape[1], self.units)

    def call(self, x):
        y = T.dot(x, self.kernel)+(self.bias.dimshuffle(('x','x',0)))
        if self.activation:
            y = self.activation(y)
        return y
    """
    def call(self, x):
        outputs_info = [None]
        xr = T.transpose(x, (1, 0, 2))
        yr, _ = theano.scan(self.step, sequences=[xr], outputs_info=outputs_info,
                            non_sequences=self.non_sequences)
        y = T.transpose(yr, (1, 0, 2))
        return y  # n, z_depth, z_k
    """