import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import W, b, pair


class CharacterLSTM(Layer):
    def __init__(self, k, units,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.k = k
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2

        h_W, h_b = pair(self, (self.k+2, self.units), "h")
        h_U = W(self, (self.units, self.units), "h_U")
        f_W, f_b = pair(self, (self.units, self.units), "f")
        i_W, i_b = pair(self, (self.units, self.units), "i")
        c_W, c_b = pair(self, (self.units, self.units), "c")
        o_W, o_b = pair(self, (self.units, self.units), "o")
        self.non_sequences = [
            h_W, h_U, h_b,
            f_W, f_b,
            i_W, i_b,
            c_W, c_b,
            o_W, o_b
        ]
        h0 = b(self, (1, self.units), "h0")
        y0 = b(self, (1, self.units), "y0")
        self.initial_states = [h0, y0]
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return (input_shape[0], self.units)

    def step(self, x, h0, y0, *params):
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
        switch = T.eq(x, 0).dimshuffle((0,'x'))
        h1s = (switch * h0) + ((1 - switch) * h1)
        y1s = (switch * y0) + ((1 - switch) * y1)
        return h1s, y1s

    def call(self, x):
        xr = T.transpose(x, (1, 0))
        n = x.shape[0]
        outputs_info = [T.extra_ops.repeat(self.initial_states[0], n, axis=0),
                        T.extra_ops.repeat(self.initial_states[1], n, axis=0)]
        (hr, yr), _ = theano.scan(self.step, sequences=xr, outputs_info=outputs_info,
                                  non_sequences=self.non_sequences)
        y = T.transpose(yr, (1, 0, 2))
        return y[:, -1, :]