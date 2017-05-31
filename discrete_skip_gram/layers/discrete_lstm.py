import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import build_kernel, build_bias, build_pair


class DiscreteLSTM(Layer):
    def __init__(self, k, units,
                 return_sequences=True,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.k = k
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        self.return_sequences = return_sequences
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2

        h_W, h_b = build_pair(self, (self.k, self.units), "h")
        h_U = build_kernel(self, (self.units, self.units), "h_U")
        f_W, f_b = build_pair(self, (self.units, self.units), "f")
        i_W, i_b = build_pair(self, (self.units, self.units), "i")
        c_W, c_b = build_pair(self, (self.units, self.units), "c")
        o_W, o_b = build_pair(self, (self.units, self.units), "o")
        self.non_sequences = [
            h_W, h_U, h_b,
            f_W, f_b,
            i_W, i_b,
            c_W, c_b,
            o_W, o_b
        ]
        self.h0 = build_bias(self, (1, self.units), "h0")
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        if self.return_sequences:
            return input_shape[0], input_shape[1], self.units
        else:
            return input_shape[0], self.units

    def step(self, x, h0, *params):
        (h_W, h_U, h_b,
         f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b) = params
        h = T.tanh(h_W[x, :] + T.dot(h0, h_U) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        y1 = (o * h1)
        return h1, y1

    def call(self, x):
        xr = T.transpose(x, (1, 0))
        n = x.shape[0]
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0), None]
        (hr, yr), _ = theano.scan(self.step, sequences=xr, outputs_info=outputs_info,
                                  non_sequences=self.non_sequences)
        y = T.transpose(yr, (1, 0, 2))
        if self.return_sequences:
            return y
        else:
            return y[:, -1, :]
