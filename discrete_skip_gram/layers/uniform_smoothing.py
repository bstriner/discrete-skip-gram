import numpy as np
from keras.layers import Layer


class UniformSmoothing(Layer):
    def __init__(self, factor=1e-8):
        self.factor = np.float32(factor)
        super(Layer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return input_shape

    def call(self, x):
        scale = 1. - (x.shape[-1] * self.factor)
        return self.factor + (scale * x)
