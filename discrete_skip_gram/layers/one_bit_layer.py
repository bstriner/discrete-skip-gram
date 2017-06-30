from theano import tensor as T

from keras import initializers
from keras import regularizers
from keras.layers import Layer
from .utils import softmax_nd, uniform_smoothing


class OneBitLayer(Layer):
    def __init__(self, y_k, z_k,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='random_uniform',
                 bias_regularizer=None
                 ):
        super(OneBitLayer, self).__init__()
        self.y_k = y_k
        self.z_k = z_k
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.supports_masking = False

    def build(self, (p_z_given_x, y)):
        y_k = self.y_k
        z_k = self.z_k
        assert len(y) == 2
        assert len(p_z_given_x) == 2
        assert p_z_given_x[1] == z_k
        assert y[1] == 1
        self.y_bias = self.add_weight(initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      shape=(y_k,), name="y_bias")
        self.y_x_bias = self.add_weight(initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        shape=(z_k, y_k), name="y_x_bias")
        self.built = True

    def compute_output_shape(self, (p_z_x, y)):
        assert len(p_z_x) == 2
        assert len(y) == 2
        return [y, y, y]

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

        # p(y|z)
        p_y_given_z = T.transpose(uniform_smoothing(softmax_nd(self.y_x_bias)), (1, 0))  # (y_k, z_k)
        p_y_given_z_t = p_y_given_z[y, :]  # (n, z_k)

        # loss = p(z|x) * log(p(y|z))
        eps = 1e-8
        prior_nll = -T.log(p_y_t)
        v2 = True
        if v2:
            posterior_nll = T.sum(p_z_given_x * -T.log(p_y_given_z_t), axis=1)  # (n,)
        else:
            posterior_nll = -T.log(eps + T.sum(p_z_given_x * p_y_given_z_t, axis=1))  # (n,)

        z_samples = T.argmax(p_z_given_x, axis=1)  # (n,)
        val_nll = -T.log(p_y_given_z[y, z_samples])  # (n,)
        prior_nll = T.reshape(prior_nll, (-1, 1))  # (n,1)
        posterior_nll = T.reshape(posterior_nll, (-1, 1))  # (n,1)
        val_nll = T.reshape(val_nll, (-1, 1))  # (n,1)

        return [prior_nll, posterior_nll, val_nll]
