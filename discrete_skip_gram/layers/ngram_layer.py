import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import W, b, pair, shift_tensor

class NgramLayer(Layer):
    """
    Given a flattened representation of x, encode as a series of symbols
    """

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
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        x = input_shape[1]
        assert (len(z) == 2)
        assert (len(x) == 2)
        input_dim = z[1]

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

    def step(self, xprev, x, h0, z, *params):
        (h_W, h_U, h_V, h_b,
         f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b,
         t_W, t_b,
         y_W, y_b) = params
        h = T.tanh(h_W[xprev, :] + T.dot(h0, h_U) + T.dot(z, h_V) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(o * h1, t_W) + t_b)
        p1 = T.nnet.softmax(T.dot(t, y_W) + y_b)
        nll1 = -T.log(p1[T.arange(p1.shape[0]), x])
        return h1, nll1

    def call(self, (z, x)):
        # z: input context: n, input_dim
        # x: ngram: n, depth int32
        xr = T.transpose(x, (1,0))
        xshifted = shift_tensor(x)
        xshiftedr = T.transpose(xshifted, (1,0))
        n = z.shape[0]
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0),
                        None]
        (hr, nllr), _ = theano.scan(self.step, sequences=[xshiftedr, xr], outputs_info=outputs_info,
                                      non_sequences=[z] + self.non_sequences)
        nll = T.transpose(nllr, (1, 0))
        return nll