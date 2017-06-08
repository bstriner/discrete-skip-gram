import theano.tensor as T

from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer
from .utils import build_embedding, sigmoid_smoothing


class SequentialEmbeddingSimpleBinary(Layer):
    """
    Shift dim 2 and pad with learned bias
    """

    def __init__(self,
                 x_k,
                 z_depth,
                 embeddings_initializer='random_uniform', embeddings_regularizer=None):
        self.x_k = x_k
        self.z_depth = z_depth
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.input_spec = [InputSpec(ndim=2)]
        self.supports_masking = False
        self.embedding = None
        super(SequentialEmbeddingSimpleBinary, self).__init__()

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return [mask, mask]

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == 1
        self.embedding = build_embedding(self, (self.x_k, self.z_depth), "embedding")
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == 1
        return [(input_shape[0], self.z_depth), (input_shape[0], self.z_depth)]

    def call(self, inputs, **kwargs):
        h = self.embedding[T.flatten(inputs), :]  # (n, z_depth)
        h = T.nnet.sigmoid(h)
        p_z = sigmoid_smoothing(h)
        z = T.gt(p_z, 0.5)  # (n, z_depth)
        return [p_z, z]
