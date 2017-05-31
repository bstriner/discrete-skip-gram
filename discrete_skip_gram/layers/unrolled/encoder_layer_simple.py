import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from ..utils import build_kernel, build_bias, build_pair
from theano.tensor.shared_randomstreams import RandomStreams

class EncoderLayerSimple(Layer):
    def __init__(self,units, z_k,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None, **kwargs):
        self.units = units
        self.z_k=z_k
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        Layer.__init__(self, **kwargs)

    def compute_mask(self, inputs, mask=None):
        #print ("Compute mask {}".format(mask))
        return mask
    def build(self, (h0, z0, x)):
        h_dim = h0[1]
        x_dim = x[1]
        self.h_W, self.h_b = build_pair(self, (h_dim, self.units), "h")
        self.h_U = build_kernel(self, (self.z_k + 1, self.units), "h_U")
        self.h_V = build_kernel(self, (x_dim, self.units), "h_V")
        self.d = build_pair(self, (self.units, self.units), "d")
        self.y1 = build_pair(self, (self.units, self.units), "y1")
        self.y2 = build_pair(self, (self.units, self.z_k), "y2")
        self.built = True

    def compute_output_shape(self, (h0, z0, x)):
        return [(h0[0], self.units), (h0[0], self.z_k)]

    def call(self, (h0, z0, x)):
        act = lambda _x: T.nnet.relu(_x, 0.2)
        t1 = act(T.dot( h0,self.h_W)+self.h_U[T.flatten(z0),:]+T.dot(x,self.h_V)+self.h_b)
        h1 = act(T.dot(t1, self.d[0]) + self.d[1])
        t2 = act(T.dot(h1, self.y1[0])+self.y1[1])
        p1 = T.nnet.softmax(T.dot(t2, self.y2[0]+self.y2[1]))
        return [h1, p1]
