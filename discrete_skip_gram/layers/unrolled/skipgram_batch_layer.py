import theano
import theano.tensor as T
from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer

from ..utils import W, b, pair, shift_tensor, softmax_nd


class SkipgramBatchLayer(Layer):
    """
    Given a flattened context, calculate NLL of a series
    """

    def __init__(self, y_k, z_k, units, mean=True,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.y_k = y_k
        self.z_k = z_k
        self.units = units
        self.mean = mean
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]
        self.supports_masking = False
        Layer.__init__(self)

    def build_params(self, input_dim):
        h_W, h_b = pair(self, (self.y_k + 1, self.units), "h")
        h_U = W(self, (self.units, self.units), "h_U")
        h_V = W(self, (input_dim, self.units), "h_V")
        f_W, f_b = pair(self, (self.units, self.units), "f")
        i_W, i_b = pair(self, (self.units, self.units), "i")
        c_W, c_b = pair(self, (self.units, self.units), "c")
        o_W, o_b = pair(self, (self.units, self.units), "o")
        t_W, t_b = pair(self, (self.units, self.units), "t")
        y_W, y_b = pair(self, (self.units, self.y_k * self.z_k), "y")
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
            return (x[0], self.z_k)
        else:
            return (x[0], self.z_k, x[1])

    def step(self, y0, y1, h0, z, *params):
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
        t1 = T.tanh(T.dot(o * h1, t_W) + t_b)
        t2 = T.dot(t1, y_W) + y_b
        t3 = T.reshape(t2, (-1, self.z_k, self.y_k))
        p1 = softmax_nd(t3)
        nll1 = -T.log(p1[T.arange(p1.shape[0]), :, y1])
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
        # nllr: y depth,n, z k
        nll = T.transpose(nllr, (1, 2, 0))  # n, z k, y
        if self.mean:
            nll = T.mean(nll, axis=2)
        return nll


class SkipgramBatchPolicyLayer(Layer):
    def __init__(self, layer, srng, depth):
        assert (isinstance(layer, SkipgramBatchLayer))
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

    def step(self, rng, h0, x0, z, znext, *params):
        (h_W, h_U, h_V, h_b,
         f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b,
         t_W, t_b,
         y_W, y_b) = params
        h = T.tanh(h_W[x0, :] + T.dot(h0, h_U) + T.dot(z, h_V) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        t1 = T.tanh(T.dot(o * h1, t_W) + t_b)
        t2 = T.dot(t1, y_W) + y_b
        t3 = T.reshape(t2, (-1, self.layer.z_k, self.layer.y_k))
        p1 = softmax_nd(t3)
        print "P1!!!!!!:{}".format(p1.ndim)
        p1 = p1[T.arange(p1.shape[0]), T.flatten(znext), :]
        c1 = T.cumsum(p1, axis=1)
        y1 = T.sum(T.gt(rng.dimshuffle((0, 'x')), c1), axis=1) + 1
        y1 = T.cast(y1, 'int32')
        return h1, y1

    def call(self, (z, znext)):
        # z: input context: n, input_dim
        n = z.shape[0]
        rngr = self.srng.uniform(low=0, high=1, dtype='float32', size=(self.depth, n))
        outputs_info = [T.extra_ops.repeat(self.layer.h0, n, axis=0),
                        T.zeros((n,), dtype='int32')]
        (hr, yr), _ = theano.scan(self.step, sequences=[rngr], outputs_info=outputs_info,
                                  non_sequences=[z, znext] + self.layer.non_sequences)
        y = T.transpose(yr, (1, 0)) - 1
        return y
