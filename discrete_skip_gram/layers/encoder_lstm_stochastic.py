import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import build_kernel, build_bias, build_pair
from theano.tensor.shared_randomstreams import RandomStreams

class EncoderLSTMStochastic(Layer):
    """
    Given a flattened representation of x, encode as a series of symbols
    """

    def __init__(self, k, units, depth,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.k = k
        self.units = units
        self.depth = depth
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

        h_W, h_b = build_pair(self, (self.k + 1, self.units), "h")
        h_U = build_kernel(self, (self.units, self.units), "h_U")
        h_V = build_kernel(self, (input_dim, self.units), "h_V")
        f_W, f_b = build_pair(self, (self.units, self.units), "f")
        i_W, i_b = build_pair(self, (self.units, self.units), "i")
        c_W, c_b = build_pair(self, (self.units, self.units), "c")
        o_W, o_b = build_pair(self, (self.units, self.units), "o")
        t_W, t_b = build_pair(self, (self.units, self.units), "t")
        y_W, y_b = build_pair(self, (self.units, self.k), "y")
        self.non_sequences = [
            h_W, h_U, h_V, h_b,
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
        self.srng = RandomStreams(123)

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        n = input_shape[0]
        return [(n, self.depth, self.k), (n, self.depth)]

    def step(self, rng, h0, y0, z, *params):
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
        c1 = T.gt(rng.dimsum(T.cumsum(p1, axis=1)
        y1 = T.argmax(p1, axis=1) + 1
        y1 = T.cast(y1, 'int32')
        y1 = theano.gradient.disconnected_grad(y1)
        return h1, p1, y1

    def call(self, z):
        n = z.shape[0]
        rng = self.srng.uniform(low=0,high=1, size=(self.depth, n), dtype='float32')
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0),
                        None,
                        T.zeros((n,), dtype='int32')]
        (hr, pr, yr), _ = theano.scan(self.step, outputs_info=outputs_info, sequences=[rng],
                                      non_sequences=[z] + self.non_sequences)
        p = T.transpose(pr, (1, 0, 2))
        y = T.transpose(yr, (1, 0)) - 1
        return [p, y]
