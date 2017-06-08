from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer
from .utils import build_embedding, uniform_smoothing, softmax_nd


class PriorLayer(Layer):
    """
    Shift dim 2 and pad with learned bias
    """

    def __init__(self,
                 x_k,
                 embeddings_initializer='random_uniform', embeddings_regularizer=None):
        self.x_k = x_k
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.input_spec = [InputSpec(ndim=2)]
        self.supports_masking = False
        self.embedding = None
        super(PriorLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == 1
        self.embedding = build_embedding(self, (self.x_k,), "embedding")
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == 1
        return input_shape

    def call(self, inputs, **kwargs):
        p_x = uniform_smoothing(softmax_nd(self.embedding))  # (x_k,)
        p_x_t = p_x[inputs]  # (n,1)
        return p_x_t
