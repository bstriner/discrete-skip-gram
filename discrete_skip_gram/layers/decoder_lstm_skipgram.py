import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import build_kernel, build_bias, build_pair
from .utils import shift_tensor, softmax_nd


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
        #input_dim = z[1]

        h_outer_W, h_outer_b = build_pair(self, (self.z_k + 1, self.units), "h_outer")
        h_outer_U = build_kernel(self, (self.units, self.units), "h_outer_U")
        f_outer_W, f_outer_b = build_pair(self, (self.units, self.units), "f_outer")
        i_outer_W, i_outer_b = build_pair(self, (self.units, self.units), "i_outer")
        c_outer_W, c_outer_b = build_pair(self, (self.units, self.units), "c_outer")
        o_outer_W, o_outer_b = build_pair(self, (self.units, self.units), "o_outer")
        t_outer_W, t_outer_b = build_pair(self, (self.units, self.units), "t_outer")
        y_outer_W, y_outer_b = build_pair(self, (self.units, self.units), "y_outer")
        h0_outer = build_bias(self, (1, self.units), "h0_outer")

        h_W, h_b = build_pair(self, (self.y_k + 1, self.units), "h")
        h_U = build_kernel(self, (self.units, self.units), "h_U")
        h_V = build_kernel(self, (self.units, self.units), "h_V")
        f_W, f_b = build_pair(self, (self.units, self.units), "f")
        i_W, i_b = build_pair(self, (self.units, self.units), "i")
        c_W, c_b = build_pair(self, (self.units, self.units), "c")
        o_W, o_b = build_pair(self, (self.units, self.units), "o")
        t_W, t_b = build_pair(self, (self.units, self.units), "t")
        y_W, y_b = build_pair(self, (self.units, self.z_k * self.y_k), "y")
        h0 = build_bias(self, (1, self.units), "h0")

        self.non_sequences = [
            h_outer_W, h_outer_U, h_outer_b,
            f_outer_W, f_outer_b,
            i_outer_W, i_outer_b,
            c_outer_W, c_outer_b,
            o_outer_W, o_outer_b,
            t_outer_W, t_outer_b,
            y_outer_W, y_outer_b,
            h_W, h_U, h_V, h_b,
            f_W, f_b,
            i_W, i_b,
            c_W, c_b,
            o_W, o_b,
            t_W, t_b,
            y_W, y_b
        ]

        self.h0_outer = h0_outer
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

    def step_internal(self, yprev, y, h0, zh, *params):
        (h_W, h_U, h_V, h_b,
        f_W, f_b,
        i_W, i_b,
        c_W, c_b,
        o_W, o_b,
        t_W, t_b,
        y_W, y_b
        ) = params
        h = T.tanh(h_W[yprev,:] + T.dot(h0, h_U) + T.dot(zh, h_V) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W)+f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W)+i_b)
        c = T.tanh(T.dot(h, c_W)+c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W)+o_b)
        h1 = (h0*f) + (c*i)
        t1 = T.tanh(T.dot(o*h1, t_W)+t_b)
        t2 = T.dot(t1, y_W)+y_b
        t3 = T.reshape(t2, (t2.shape[0], self.z_k, self.y_k))
        py1 = softmax_nd(t3)
        py2 = py1[T.arange(py1.shape[0]), :, y]
        nll = -T.log(py2) #n, z_k
        return h1, nll

    def step(self, zprev, h0, yprev, y, hinit, *params):
        (h_outer_W, h_outer_U, h_outer_b,
            f_outer_W, f_outer_b,
            i_outer_W, i_outer_b,
            c_outer_W, c_outer_b,
            o_outer_W, o_outer_b,
            t_outer_W, t_outer_b,
            y_outer_W, y_outer_b,
            h_W, h_U, h_V, h_b,
            f_W, f_b,
            i_W, i_b,
            c_W, c_b,
            o_W, o_b,
            t_W, t_b,
            y_W, y_b
        ) = params

        h = T.tanh(h_outer_W[zprev, :] + T.dot(h0, h_outer_U) + h_outer_b)
        f = T.nnet.sigmoid(T.dot(h, f_outer_W) + f_outer_b)
        i = T.nnet.sigmoid(T.dot(h, i_outer_W) + i_outer_b)
        c = T.tanh(T.dot(h, c_outer_W) + c_outer_b)
        o = T.nnet.sigmoid(T.dot(h, o_outer_W) + o_outer_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(o * h1, t_outer_W) + t_outer_b)
        zh = T.tanh(T.dot(t, y_outer_W) + y_outer_b)

        n = zprev.shape[0]
        outputs_info = [T.extra_ops.repeat(hinit, n, axis=0),
                        None]
        inner_non_sequences = [
            h_W, h_U, h_V, h_b,
            f_W, f_b,
            i_W, i_b,
            c_W, c_b,
            o_W, o_b,
            t_W, t_b,
            y_W, y_b
        ]
        hinner, nll = theano.scan(self.step_internal,
                                  sequences=[yprev, y],
                                  outputs_info=outputs_info,
                                  non_sequences=[zh]+inner_non_sequences,
                                  )
        #nll: y_depth, n, z_k
        nllt = T.sum(nll, axis=0) #n, z_k

        return h1, nllt

    def call(self, (z, y)):
        # z: n, z_depth (int) encoding
        # y: n, y_depth (int) actual output
        # output: n, z_depth, y_depth (float32) NLL
        n = z.shape[0]
        zprev = shift_tensor(z)
        zprevr = T.transpose(zprev, (1, 0))
        #zr = T.transpose(z, (1, 0))
        yprev = shift_tensor(y)
        yprevr = T.transpose(yprev, (1, 0))
        yr = T.transpose(y, (1,0))
        outputs_info = [T.extra_ops.repeat(self.h0_outer, n, axis=0),
                        None,
                        T.zeros((n,), dtype='int32')]
        (hr, nllr), _ = theano.scan(self.step, sequences=[zprevr], outputs_info=outputs_info,
                                      non_sequences=[yprevr, yr, self.h0] + self.non_sequences)
        #nllr: z_depth, n, z_k
        nll = T.transpose(nllr, (1,0,2))
        return nll # n, z_depth, z_k


class DecoderLSTMSkipgramPolicy(Layer):
    """
    Given a flattened representation of x, encode as a series of symbols
    """

    def __init__(self, layer):
        self.layer=layer
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        rng = input_shape[1]
        assert (len(z) == 2)
        assert (len(rng) == 2)
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
        return (z[0], self.layer.z_k, self.layer.x_k)

    def step_internal(self, z, rng, h0, y0, zh, *params):
        (h_W, h_U, h_V, h_b,
        f_W, f_b,
        i_W, i_b,
        c_W, c_b,
        o_W, o_b,
        t_W, t_b,
        y_W, y_b
        ) = params
        h = T.tanh(h_W[y0,:] + T.dot(h0, h_U) + T.dot(zh, h_V) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W)+f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W)+i_b)
        c = T.tanh(T.dot(h, c_W)+c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W)+o_b)
        h1 = (h0*f) + (c*i)
        t1 = T.tanh(T.dot(o*h1, t_W)+t_b)
        t2 = T.dot(t1, y_W)+y_b
        t3 = T.reshape(t2, (t2.shape[0], self.z_k, self.y_k))
        py1 = softmax_nd(t3) #n, z_k, y_k
        py2 = py1[T.arange(py1.shape[0]), z, :] #n, y_k
        cs1 = T.cumsum(py2, axis=1)
        y1 = T.sum(T.gt(rng.dimshuffle(('x',0)), cs1), axis=1)+1
        y1 = T.cast(y1, 'int32') # (n,)
        return h1, y1

    def step(self, zprev, z, h0, rng, hinit, *params):
        (h_outer_W, h_outer_U, h_outer_b,
            f_outer_W, f_outer_b,
            i_outer_W, i_outer_b,
            c_outer_W, c_outer_b,
            o_outer_W, o_outer_b,
            t_outer_W, t_outer_b,
            y_outer_W, y_outer_b,
            h_W, h_U, h_V, h_b,
            f_W, f_b,
            i_W, i_b,
            c_W, c_b,
            o_W, o_b,
            t_W, t_b,
            y_W, y_b
        ) = params

        h = T.tanh(h_outer_W[zprev, :] + T.dot(h0, h_outer_U) + h_outer_b)
        f = T.nnet.sigmoid(T.dot(h, f_outer_W) + f_outer_b)
        i = T.nnet.sigmoid(T.dot(h, i_outer_W) + i_outer_b)
        c = T.tanh(T.dot(h, c_outer_W) + c_outer_b)
        o = T.nnet.sigmoid(T.dot(h, o_outer_W) + o_outer_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(o * h1, t_outer_W) + t_outer_b)
        zh = T.tanh(T.dot(t, y_outer_W) + y_outer_b)

        n = zprev.shape[0]
        outputs_info = [T.extra_ops.repeat(hinit, n, axis=0),
                        T.zeros((n,), dtype='int32')]
        inner_non_sequences = [
            h_W, h_U, h_V, h_b,
            f_W, f_b,
            i_W, i_b,
            c_W, c_b,
            o_W, o_b,
            t_W, t_b,
            y_W, y_b
        ]
        hinner, yr = theano.scan(self.step_internal,
                                  sequences=[z, rng],
                                  outputs_info=outputs_info,
                                  non_sequences=[zh]+inner_non_sequences,
                                  )
        #yr: y_depth, n

        return h1, yr

    def call(self, (z, rng)):
        # z: n, z_depth (int) encoding
        # rng: n, y_depth (float) rng
        # output: n, z_depth, y_depth (float32) NLL
        n = z.shape[0]
        zprev = shift_tensor(z)
        zprevr = T.transpose(zprev, (1, 0))
        zr = T.transpose(z, (1, 0))
        rngr = T.transpose((rng, (1,0)))
        outputs_info = [T.extra_ops.repeat(self.layer.h0_outer, n, axis=0),
                        None,
                        T.zeros((n,), dtype='int32')]
        (hr, yr), _ = theano.scan(self.step, sequences=[zprevr, zr], outputs_info=outputs_info,
                                      non_sequences=[rngr, self.layer.h0] + self.layer.non_sequences)
        #yr: z_depth, y_depth, n
        y = T.transpose(yr, (2,0,1))
        return y # n, z_depth, y_depth
