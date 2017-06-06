import numpy as np

from keras.layers import Layer
from .utils import uniform_smoothing


class UniformSmoothing(Layer):
    def __init__(self, factor=1e-9, **kwargs):
        self.factor = np.float32(factor)
        super(UniformSmoothing, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) >= 2
        return input_shape

    def call(self, x):
        return uniform_smoothing(x, factor=self.factor)
