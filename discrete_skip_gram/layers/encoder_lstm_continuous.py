import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import build_kernel, build_bias, build_pair
from keras import activations

class EncoderLSTMContinuous(Layer):
    """
    Given a flattened representation of x, encode as a series of vectors
    """

    def __init__(self, z_depth, z_k, units,
                 activation=T.tanh,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None,
                 activity_regularizer=None):
        self.z_depth = z_depth
        self.z_k = z_k
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        h_W, h_b = build_pair(self, (input_dim, self.units), "h")
        h_U = build_kernel(self, (self.units, self.units), "h_U")
        f_W, f_b = build_pair(self, (self.units, self.units), "f")
        i_W, i_b = build_pair(self, (self.units, self.units), "i")
        c_W, c_b = build_pair(self, (self.units, self.units), "c")
        o_W, o_b = build_pair(self, (self.units, self.units), "o")
        t_W, t_b = build_pair(self, (self.units, self.units), "t")
        y_W, y_b = build_pair(self, (self.units, self.z_k), "y")
        self.non_sequences = [
            h_W, h_U, h_b,
            f_W, f_b,
            i_W, i_b,
            c_W, c_b,
            o_W, o_b,
            t_W, t_b,
            y_W, y_b
        ]
        h0 = build_bias(self, (1, self.units), "h0")
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
        y1 = self.activation(T.dot(t, y_W) + y_b)
        return h1, y1

    def call(self, x):
        n = x.shape[0]
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0),
                        None]
        (hr, yr), _ = theano.scan(self.step, outputs_info=outputs_info, n_steps=self.z_depth,
                                  non_sequences=[x] + self.non_sequences)
        y = T.transpose(yr, (1, 0, 2))
        return y  # n, z_depth, z_k
