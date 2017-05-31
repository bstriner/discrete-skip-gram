import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import build_kernel, build_bias, build_pair



class DecoderMerge(Layer):
    """
    Inputs:
        * y: series of output sequences, shifted+1
        * dec_h: series of encodings
    """

    def __init__(self, y_k, z_k, units,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.y_k = y_k
        self.z_k = z_k
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        Layer.__init__(self)

    def build(self, input_shape):
        print "Merge shapes: {}".format(input_shape)
        assert len(input_shape) == 2
        y = input_shape[0]
        dec_h = input_shape[1]
        assert (len(y) == 3)
        assert (len(dec_h) == 3)
        input_dim_y = y[2]
        input_dim_h = dec_h[2]

        h1_W, h1_b = build_pair(self, (input_dim_y, self.units), "h1")
        h1_U = build_kernel(self, (input_dim_h, self.units), "h1_U")
        h2_W, h2_b = build_pair(self, (self.units, self.units), "h2")
        y_W, y_b = build_pair(self, (self.units, self.z_k * (self.y_k + 2)), "y")
        self.non_sequences = [
            h1_W, h1_U, h1_b,
            h2_W, h2_b,
            y_W, y_b
        ]
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        hy = input_shape[0]
        dec_h = input_shape[1]
        assert (len(hy) == 3)
        assert (len(dec_h) == 3)
        return dec_h[0], dec_h[1], hy[1], self.z_k, self.y_k + 2

    def inner_step(self, hy, dec_h, *params):
        (h1_W, h1_U, h1_b,
         h2_W, h2_b,
         y_W, y_b) = params
        h1 = T.tanh(T.dot(hy, h1_W) + T.dot(dec_h, h1_U) + h1_b)
        h2 = T.tanh(T.dot(h1, h2_W) + h2_b)
        y1 = T.reshape(T.dot(h2, y_W) + y_b, (h2.shape[0], self.z_k, self.y_k + 2))
        y1 = softmax_2d(y1)
        return y1

    def step(self, dec_h, hy, *params):
        outputs_info = [None]
        y1, _ = theano.scan(self.inner_step, sequences=[hy], outputs_info=outputs_info,
                            non_sequences=[dec_h] + list(params))
        return y1

    def call(self, (hy, dec_h)):
        """
        y: lstm output of shifted actual targets
        dec_h: shifted discrete encoding
        output: (n, encoding_k, y_k)
        :return:
        """
        hyr = T.transpose(hy, (1, 0, 2))
        dec_h_r = T.transpose(dec_h, (1, 0, 2))
        outputs_info = [None]
        outputr, _ = theano.scan(self.step, sequences=dec_h_r, outputs_info=outputs_info,
                                 non_sequences=[hyr] + self.non_sequences)
        # outputr: z_depth, x_depth, n, z_k, x_k
        # output: n, z depth, x depth, z_k, x_k
        output = T.transpose(outputr, (2, 0, 1, 3, 4))
        return output
