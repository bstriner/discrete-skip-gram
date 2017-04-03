import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import W, b, pair
from keras import backend as K
from .utils import shift_tensor
from theano.tensor.shared_randomstreams import RandomStreams

class DQNEncoderValue(Layer):
    """
    Value function for encoding
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
        x = input_shape[0]
        z = input_shape[1]
        assert (len(x) == 2)
        assert (len(z) == 2)
        input_dim = x[1]

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

    #def compute_mask(self, inputs, mask=None):
    #    print ("Compute mask {}".format(mask))
     #   return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        x = input_shape[0]
        z = input_shape[1]
        assert (len(x) == 2)
        assert (len(z) == 2)
        return z

    def step(self, zprev, z, h0, x, *params):
        (h_W, h_U, h_V, h_b,
         f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b,
         t_W, t_b,
         y_W, y_b) = params
        h = T.tanh(h_W[zprev, :] + T.dot(h0, h_U) + T.dot(x, h_V) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(o * h1, t_W) + t_b)
        vals1 = T.dot(t, y_W) + y_b
        v1 = vals1[T.arange(vals1.shape[0]), z]
        return h1, v1

    def call(self, (x, z)):
        """
        x: context of input
        z: series of inputs
        :return: 
        """
        zprev = shift_tensor(z)
        zprevr = T.transpose(zprev, (1,0))
        zr = T.transpose(z, (1, 0))
        n = x.shape[0]
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0),
                        None]
        (hr, vr), _ = theano.scan(self.step, sequences=[zprevr, zr],
                                      outputs_info=outputs_info,
                                      non_sequences=[x] + self.non_sequences)
        v = T.transpose(vr, (1, 0))
        return v

class DQNEncoderPolicy(Layer):
    """
    Value function for encoding
    """

    def __init__(self, value_layer, depth, exploration):
        self.value_layer = value_layer
        self.depth = depth
        self.exploration = K.variable(exploration, dtype='float32', name="exploration")
        self.srng = RandomStreams(123)
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return input_shape[0], self.depth

    def step(self, rngf, rngi, h0, z0, x, exploration, *params):
        (h_W, h_U, h_V, h_b,
         f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b,
         t_W, t_b,
         y_W, y_b) = params
        h = T.tanh(h_W[z0, :] + T.dot(h0, h_U) + T.dot(x, h_V) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(o * h1, t_W) + t_b)
        vals1 = T.dot(t, y_W) + y_b
        z1 = T.argmax(vals1, axis=1)
        switch = T.gt(exploration, rngf)
        z1 = (switch*rngi)+((1-switch)*z1)+1
        z1 = T.cast(z1, 'int32')
        return h1, z1

    def call(self, x, stochastic=True):
        """
        x: context of input
        :return: 
        """
        n = x.shape[0]
        if stochastic:
            rngf = self.srng.uniform(size=(self.depth, n),
                                     low=0,
                                     high=1,
                                     dtype='float32')
        else:
            rngf = T.ones((self.depth, n), dtype='float32')
        rngi = self.srng.random_integers(size=(self.depth, n),
                                         low=0,
                                         high=self.value_layer.k-1,
                                         dtype='int32')

        outputs_info = [T.extra_ops.repeat(self.value_layer.h0, n, axis=0),
                        T.zeros((n,), dtype='int32')]
        (hr, zr), _ = theano.scan(self.step, sequences=[rngf, rngi],
                                  outputs_info=outputs_info,
                                  non_sequences=[x, self.exploration] + self.value_layer.non_sequences)
        z = T.transpose(zr, (1, 0)) - 1
        return z
