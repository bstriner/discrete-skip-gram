import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from ..utils import W, b, pair, shift_tensor


class SkipgramLayer(Layer):
    """
    Given a flattened context, calculate NLL of a series
    """

    def __init__(self, k, units, embedding_units, mean=True,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.k = k
        self.units = units
        self.embedding_units = embedding_units
        self.mean = mean
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]
        self.supports_masking = False
        Layer.__init__(self)

    def build_params(self, input_dim):
        h_W, h_b = pair(self, (self.units, self.units), "h")
        h_U1 = W(self, (self.k + 1, self.embedding_units), "h_U1")
        h_U2 = W(self, (self.embedding_units, self.units), "h_U2")
        h_V = W(self, (input_dim, self.units), "h_V")
        f_W, f_b = pair(self, (self.units, self.units), "f")
        i_W, i_b = pair(self, (self.units, self.units), "i")
        c_W, c_b = pair(self, (self.units, self.units), "c")
        o_W, o_b = pair(self, (self.units, self.units), "o")
        t_W, t_b = pair(self, (self.units, self.units), "t")
        y_W, y_b = pair(self, (self.units, self.k), "y")
        self.non_sequences = [
            h_W, h_U1, h_U2, h_V, h_b,
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

    def build(self, (z, x)):
        assert (len(z) == 2)
        assert (len(x) == 2)
        input_dim = z[1]
        self.build_params(input_dim)

    #    def compute_mask(self, inputs, mask=None):
    #        print ("Compute mask {}".format(mask))
    #        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        x = input_shape[1]
        assert (len(z) == 2)
        assert (len(x) == 2)
        if self.mean:
            return (x[0], 1)
        else:
            return x

    def step(self, y0, y1, h0, z, *params):
        (h_W, h_U1, h_U2, h_V, h_b,
         f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b,
         t_W, t_b,
         y_W, y_b) = params
        h = T.tanh(T.dot(h0,h_W) + T.dot(h_U1[y0,:], h_U2) + T.dot(z, h_V) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(o * h1, t_W) + t_b)
        # print "NDim sg"
        # print y0.ndim
        # print y1.ndim
        # print h0.ndim
        # print z.ndim
        # print h.ndim
        # print h1.ndim
        # print t.ndim
        p1 = T.nnet.softmax(T.dot(t, y_W) + y_b)
        nll1 = -T.log(p1[T.arange(p1.shape[0]), y1])
        # nll1 = T.reshape(nll1,(-1,1))
        return h1, nll1

    def call(self, (z, y)):
        # z: input context: n, input_dim
        # y: ngram: n, depth int32
        # print "Z NDIM: {}".format(z.ndim)
        yr = T.transpose(y, (1, 0))
        yshifted = shift_tensor(y)
        yshiftedr = T.transpose(yshifted, (1, 0))
        n = z.shape[0]
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0),
                        None]
        (hr, nllr), _ = theano.scan(self.step, sequences=[yshiftedr, yr], outputs_info=outputs_info,
                                    non_sequences=[z] + self.non_sequences)
        nll = T.transpose(nllr, (1, 0))
        if self.mean:
            nll = T.mean(nll, axis=1, keepdims=True)
        return nll


class SkipgramPolicyLayer(Layer):
    def __init__(self, layer, srng, depth):
        self.layer = layer
        self.srng = srng
        self.depth = depth
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.built = True

    def compute_mask(self, inputs, mask=None):
        # print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return (input_shape[0], self.depth)

    def step(self, rng, h0, y0, z, *params):
        (h_W, h_U1, h_U2, h_V, h_b,
         f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b,
         t_W, t_b,
         y_W, y_b) = params
        h = T.tanh(T.dot(h0,h_W) + T.dot(h_U1[y0,:], h_U2) + T.dot(z, h_V) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(o * h1, t_W) + t_b)
        p1 = T.nnet.softmax(T.dot(t, y_W) + y_b)
        c1 = T.cumsum(p1, axis=1)
        y1 = T.sum(T.gt(rng.dimshuffle((0, 'x')), c1), axis=1) + 1
        y1 = T.cast(y1, 'int32')
        return h1, y1

    def call(self, z):
        # z: input context: n, input_dim
        n = z.shape[0]
        rngr = self.srng.uniform(low=0, high=1, dtype='float32', size=(self.depth, n))
        outputs_info = [T.extra_ops.repeat(self.layer.h0, n, axis=0),
                        T.zeros((n,), dtype='int32')]
        (hr, yr), _ = theano.scan(self.step, sequences=[rngr], outputs_info=outputs_info,
                                  non_sequences=[z] + self.layer.non_sequences)
        y = T.transpose(yr, (1, 0)) - 1
        return y
