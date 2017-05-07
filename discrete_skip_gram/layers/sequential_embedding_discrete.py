import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from theano.tensor.shared_randomstreams import RandomStreams
import keras.backend as K


class SequentialEmbeddingDiscrete(Layer):
    """
    Given a flattened representation of x, encode as a discrete series of symbols.
    Does not learn.
    """

    def __init__(self, embedding):
        assert len(embedding.shape) == 2
        self.depth = embedding.shape[1]
        self.embedding = K.variable(embedding, dtype='int32')
        self.input_spec = InputSpec(ndim=2)
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.built = True

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return tuple(list(input_shape) + [self.depth])

    def call(self, x):
        return self.embedding[x, :]
