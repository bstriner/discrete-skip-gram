import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import W, b, pair


class TimeDistributedDense(Layer):
    """
    
    """

    def __init__(self, units,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.units = units
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

        h_W, h_b = pair(self, (input_dim, self.units), "h")
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

    def step(self, x, *params):
        (h_W, h_b) = params
        y = T.dot(x, h_W) + h_b
        return y

    def call(self, x):
        outputs_info = [None]
        xr = T.transpose(x, (1, 0, 2))
        yr, _ = theano.scan(self.step, sequences=[xr], outputs_info=outputs_info,
                              non_sequences=self.non_sequences)
        y = T.transpose(yr, (1, 0, 2))
        return y  # n, z_depth, z_k
