import theano.tensor as T
from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer

from ..utils import W, pair


class DecoderLayerSimple(Layer):
    def __init__(self, units, z_k,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None, **kwargs):
        self.units = units
        self.z_k = z_k
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        Layer.__init__(self, **kwargs)

    def build(self, (h0, z)):
        h_dim = h0[1]
        self.h_W, self.h_b = pair(self, (h_dim, self.units), "h")
        self.h_U = W(self, (self.z_k, self.units), "h_U")
        self.d = pair(self, (self.units, self.units), "d")
        self.y1 = pair(self, (self.units, self.units), "y1")
        self.y2 = pair(self, (self.units, self.units), "y2")
        self.built = True

    def compute_mask(self, inputs, mask=None):
        #print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.units), (input_shape[0], self.units)]

    def call(self, (h0, z)):
        act = lambda _x:T.nnet.relu(_x,0.2)
        tmp1 = act(T.dot(h0, self.h_W) + self.h_U[T.flatten(z), :] + self.h_b)
        h1 = act(T.dot(tmp1, self.d[0])+self.d[1])
        tmp2 = act(T.dot(h1, self.y1[0])+self.y1[1])
        zh = act(T.dot(tmp2, self.y2[0])+self.y2[1])
        return [h1, zh]
