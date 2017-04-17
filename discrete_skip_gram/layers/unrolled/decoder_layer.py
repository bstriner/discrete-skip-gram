import theano.tensor as T
from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer

from ..utils import W, pair


class DecoderLayer(Layer):
    def __init__(self, units, z_k,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.units = units
        self.z_k = z_k
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        Layer.__init__(self)

    def build(self, (h0, z)):
        h_dim = h0[1]
        self.h_W, self.h_b = pair(self, (h_dim, self.units), "h")
        self.h_U = W(self, (self.z_k, self.units), "h_U")
        self.f = pair(self, (self.units, self.units), "f")
        self.i = pair(self, (self.units, self.units), "i")
        self.c = pair(self, (self.units, self.units), "c")
        self.o = pair(self, (self.units, self.units), "o")
        self.t = pair(self, (self.units, self.units), "t")
        self.y = pair(self, (self.units, self.units), "y")
        self.built = True

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.units), (input_shape[0], self.units)]

    def call(self, (h0, z)):
        h = T.tanh(T.dot(h0, self.h_W) + self.h_U[T.flatten(z), :] + self.h_b)
        f = T.nnet.sigmoid(T.dot(h, self.f[0]) + self.f[1])
        i = T.nnet.sigmoid(T.dot(h, self.i[0]) + self.i[1])
        c = T.tanh(T.dot(h, self.c[0]) + self.c[1])
        o = T.nnet.sigmoid(T.dot(h, self.o[0]) + self.o[1])
        h1 = (h0 * f) + (c * i)
        t1 = T.tanh(T.dot(h1 * o, self.t[0]) + self.t[1])
        zh = T.tanh(T.dot(t1, self.y[0]) + self.y[1])
        return [h1, zh]
