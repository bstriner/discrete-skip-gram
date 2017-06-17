import keras.backend as K
from keras.engine import InputSpec
from keras.layers import Layer


class SequentialEmbeddingDiscrete(Layer):
    """
    Given a flattened representation of x, encode as a discrete series of symbols.
    Does not learn.
    """

    def __init__(self, embedding):
        super(SequentialEmbeddingDiscrete, self).__init__()
        assert len(embedding.shape) == 2
        self.depth = embedding.shape[1]
        self.initial_embedding = embedding
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.embedding = self.add_weight(name="embedding",
                                         shape=self.initial_embedding.shape,
                                         dtype='int32',
                                         initializer='zero',
                                         trainable=False)
        K.set_value(self.embedding, self.initial_embedding)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return tuple(list(input_shape) + [self.depth])

    def call(self, x):
        return self.embedding[x, :]
