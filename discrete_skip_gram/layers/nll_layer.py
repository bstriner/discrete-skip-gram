from keras.engine import InputSpec
import theano.tensor as T

from keras.engine import InputSpec
from keras.layers import Layer


class NLL(Layer):
    """
    Shift dim 2 and pad with learned bias
    """

    def __init__(self):
        self.input_spec = [InputSpec()]
        self.supports_masking = False
        super(NLL, self).__init__()

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        return -T.log(inputs)
