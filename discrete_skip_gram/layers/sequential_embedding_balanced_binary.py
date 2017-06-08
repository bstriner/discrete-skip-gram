import theano.tensor as T

from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer
from ..layers.utils import build_embedding, sigmoid_smoothing, build_bias


class SequentialEmbeddingBalancedBinary(Layer):
    """
    Shift dim 2 and pad with learned bias
    """

    def __init__(self,
                 x_k,
                 z_depth,
                 units,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None,
                 embeddings_initializer='random_uniform', embeddings_regularizer=None):
        self.x_k = x_k
        self.z_depth = z_depth
        self.units = units
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = [InputSpec(ndim=2)]
        self.supports_masking = False
        self.embedding = None
        self.h0 = None
        super(SequentialEmbeddingBalancedBinary, self).__init__()

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return [mask, mask]

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == 1
        self.embedding = build_embedding(self, (self.x_k, self.z_depth), "embedding")
        self.h0 = build_bias(self, (self.units,), "h0")
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == 1
        return [(input_shape[0], self.z_depth), (input_shape[0], self.z_depth)]

    def call(self, inputs, **kwargs):
        p = sigmoid_smoothing(T.nnet.sigmoid(self.embedding))  # (x_k, z_depth)
        pr = T.transpose(p, (1, 0))  # z_depth, x_k

        h = self.embedding[T.flatten(inputs), :]  # (n, z_depth)
        h = T.nnet.sigmoid(h)
        p_z = sigmoid_smoothing(h)
        z = T.gt(p_z, 0.5)  # (n, z_depth)
        return [p_z, z]
