import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import W, b, pair


class EncoderLSTMContinuous(Layer):
    """
    Given a flattened representation of x, encode as a series of vectors
    """

    def __init__(self, z_depth, z_k, units,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.z_depth = z_depth
        self.z_k = z_k
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        h_W, h_b = pair(self, (input_dim, self.units), "h")
        h_U = W(self, (self.units, self.units), "h_U")
        f_W, f_b = pair(self, (self.units, self.units), "f")
        i_W, i_b = pair(self, (self.units, self.units), "i")
        c_W, c_b = pair(self, (self.units, self.units), "c")
        o_W, o_b = pair(self, (self.units, self.units), "o")
        t_W, t_b = pair(self, (self.units, self.units), "t")
        y_W, y_b = pair(self, (self.units, self.z_k), "y")
        self.non_sequences = [
            h_W, h_U, h_b,
            f_W, f_b,
            i_W, i_b,
            c_W, c_b,
            o_W, o_b,
            t_W, t_b,
            y_W, y_b
        ]
        h0 = b(self, (1, self.units), "h0")
        self.h0 = h0
        self.built = True

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return (input_shape[0], self.z_depth, self.z_k)

    def step(self, h0, x, *params):
        (h_W, h_U, h_b,
         f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b,
         t_W, t_b,
         y_W, y_b) = params
        h = T.tanh(T.dot(x, h_W) + T.dot(h0, h_U) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(o * h1, t_W) + t_b)
        y1 = T.tanh(T.dot(t, y_W) + y_b)
        return h1, y1

    def call(self, x):
        n = x.shape[0]
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0),
                        None,
                        None]
        (hr, yr), _ = theano.scan(self.step, outputs_info=outputs_info,
                                      non_sequences=[x] + self.non_sequences)
        y = T.transpose(yr, (1, 0, 2))
        return y  #n, z_depth, z_k