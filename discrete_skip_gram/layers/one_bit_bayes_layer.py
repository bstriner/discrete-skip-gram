from keras import initializers
from keras.layers import Layer
from theano import tensor as T

from .utils import softmax_nd, uniform_smoothing


class OneBitBayesLayer(Layer):
    def __init__(self, y_k, z_k,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='random_uniform'
                 ):
        super(OneBitBayesLayer, self).__init__()
        self.y_k = y_k
        self.z_k = z_k
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.supports_masking = False

    def build(self, input_shape):
        y_k = self.y_k
        z_k = self.z_k
        self.y_bias = self.add_weight(initializer=self.bias_initializer, shape=(y_k,), name="y_bias")
        self.z_y_bias = self.add_weight(initializer=self.kernel_initializer, shape=(y_k, z_k), name="z_y_bias")
        self.built = True

    def compute_output_shape(self, (p_z_x, y)):
        assert len(p_z_x) == 2
        assert len(y) == 2
        return [y, y]

    def compute_mask(self, inputs, mask=None):
        print "Computing mask: {}".format(inputs)
        return inputs

    def call(self, (p_z_given_x, y)):
        """
        p_z_given_x: (n, z_k) p(z|x)
        y: (n, 1) int32
        :return:
        """

        y = T.flatten(y)

        # p(y)
        p_y = uniform_smoothing(softmax_nd(self.y_bias))  # (y_k,)
        p_y_t = p_y[y]  # (n,)

        # p(z|y)
        p_z_given_y = uniform_smoothing(softmax_nd(self.z_y_bias))  # (y_k, z_k)
        p_z_given_y_t = p_z_given_y[y, :]  # (n, z_k)

        # p(z)
        p_z = T.sum((p_y.dimshuffle((0, 'x'))) * p_z_given_y, axis=0)  # (z_k,)

        # p(y|x) = p(y|z)p(z|x) = p(y) p(z|y)p(z|x)/p(z)
        p_y_z = p_y_t * T.sum(p_z_given_y_t * p_z_given_x / p_z, axis=1)  # (n,)

        # nll
        prior_nll = -T.log(p_y_t)
        prior_nll = T.reshape(prior_nll, (-1, 1))
        posterior_nll = -T.log(p_y_z)
        posterior_nll = T.reshape(posterior_nll, (-1, 1))

        return [prior_nll, posterior_nll]
