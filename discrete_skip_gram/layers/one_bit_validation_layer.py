from keras import initializers
from keras.layers import Layer
from theano import tensor as T

from .utils import softmax_nd, uniform_smoothing


class OneBitValidationLayer(Layer):
    def __init__(self, y_k, z_k,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='random_uniform'
                 ):
        super(OneBitValidationLayer, self).__init__()
        self.y_k = y_k
        self.z_k = z_k
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.supports_masking = False

    def build(self, input_shape):
        y_k = self.y_k
        z_k = self.z_k
        self.y_z_bias = self.add_weight(initializer=self.kernel_initializer, shape=(z_k, y_k), name="y_z_bias")
        self.built = True

    def compute_output_shape(self, (p_z_x, y)):
        assert len(p_z_x) == 2
        assert len(y) == 2
        return y

    def compute_mask(self, inputs, mask=None):
        print "Computing mask: {}".format(inputs)
        return inputs[0]

    def call(self, (p_z_given_x, y)):
        """
        p_z_given_x: (n, z_k) p(z|x)
        y: (n, 1) int32
        :return:
        """

        z = T.argmax(p_z_given_x, axis=1)  # (n,)
        y = T.flatten(y)  # (n.)

        # p(y|z)
        p_y_given_z = uniform_smoothing(softmax_nd(self.y_z_bias))  # (z_k, y_k)
        p_y_given_z_t = p_y_given_z[z, y]
        nll = -T.log(p_y_given_z_t)
        return nll

