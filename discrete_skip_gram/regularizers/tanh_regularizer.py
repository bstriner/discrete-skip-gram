from keras.regularizers import Regularizer
import theano.tensor as T
import keras.backend as K
import numpy as np

class TanhRegularizer(Regularizer):
    def __init__(self, weight):
        self.weight=np.float32(weight)
        Regularizer.__init__(self)

    def __call__(self, x):
        return self.weight*T.sum(T.log(K.epsilon()+1-T.abs_(x)), axis=None)