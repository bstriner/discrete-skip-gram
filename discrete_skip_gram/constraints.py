from keras.constraints import Constraint
import theano.tensor as T
import numpy as np


class ClipConstraint(Constraint):
    def __init__(self, value):
        self.value = np.float32(value)

    def __call__(self, x):
        return T.clip(x, -self.value, self.value)
