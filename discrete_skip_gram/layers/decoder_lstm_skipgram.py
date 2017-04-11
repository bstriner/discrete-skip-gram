import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import W, b, pair
from .utils import shift_tensor


class DecoderLSTMSkipgram(Layer):
    """
    Given a flattened representation of x, encode as a series of symbols
    """

    def __init__(self, z_k, y_k, units,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.z_k = z_k
        self.y_k = y_k
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
        z = input_shape[0]
        rng = input_shape[1]
        assert (len(z) == 2)
        assert (len(rng) == 2)
        input_dim = z[1]

        h_outer_W, h_outer_b = pair(self, (self.z_k + 1, self.units), "h_outer")
        h_outer_U = W(self, (self.units, self.units), "h_outer_U")
        f_outer_W, f_outer_b = pair(self, (self.units, self.units), "f_outer")
        i_outer_W, i_outer_b = pair(self, (self.units, self.units), "i_outer")
        c_outer_W, c_outer_b = pair(self, (self.units, self.units), "c_outer")
        o_outer_W, o_outer_b = pair(self, (self.units, self.units), "o_outer")
        t_outer_W, t_outer_b = pair(self, (self.units, self.units), "t_outer")
        y_outer_W, y_outer_b = pair(self, (self.units, self.k), "y_outer")
        h0_outer = b(self, (1, self.units), "h0_outer")

        h_W, h_b = pair(self, (self.k + 1, self.units), "h")
        h_U = W(self, (self.units, self.units), "h_U")
        h_V = W(self, (input_dim, self.units), "h_V")
        f_W, f_b = pair(self, (self.units, self.units), "f")
        i_W, i_b = pair(self, (self.units, self.units), "i")
        c_W, c_b = pair(self, (self.units, self.units), "c")
        o_W, o_b = pair(self, (self.units, self.units), "o")
        t_W, t_b = pair(self, (self.units, self.units), "t")
        y_W, y_b = pair(self, (self.units, self.k), "y")
        self.non_sequences = [
            h_W, h_U, h_V, h_b,
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
        z = input_shape[0]
        rng = input_shape[1]
        assert (len(z) == 2)
        assert (len(rng) == 2)
        return [(z[0], self.k), (z[0], rng[1])]

    def step(self, zprev, z, h0, yprev, y, *params):
        (h_W, h_U, h_V, h_b,
         f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b,
         t_W, t_b,
         y_W, y_b) = params
        h = T.tanh(h_W[y0, :] + T.dot(h0, h_U) + T.dot(z, h_V) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(o * h1, t_W) + t_b)
        p1 = T.nnet.softmax(T.dot(t, y_W) + y_b)
        csum = T.cumsum(p1, axis=1)
        y1 = T.sum(T.gt(rng.dimshuffle((0, 'x')), csum), axis=1) + 1
        y1 = T.cast(y1, 'int32')
        y1 = theano.gradient.disconnected_grad(y1)
        return h1, p1, y1

    def call(self, (z, y)):
        # z: n, z_depth (int)
        # y: n, y_depth (int)
        # output: n, z_depth, y_depth (float32) NLL
        n = z.shape[0]
        zprev = shift_tensor(z)
        zprevr = T.transpose(zprev, (1, 0))
        zr = T.transpose(z, (1, 0))
        yprev = shift_tensor(y)
        yprevr = T.transpose(yprev, (1, 0))
        yr = T.transpose(y, (1,0))
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0),
                        None,
                        T.zeros((n,), dtype='int32')]
        (hr, pr, yr), _ = theano.scan(self.step, sequences=[zprevr, zr], outputs_info=outputs_info,
                                      non_sequences=[yprevr, yr] + self.non_sequences)
        p = T.transpose(pr, (1, 0, 2))
        y = T.transpose(yr, (1, 0)) - 1
        return [p, y]
