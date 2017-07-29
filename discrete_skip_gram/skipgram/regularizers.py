import theano.tensor as T

from keras.regularizers import Regularizer


class ExclusiveLasso(Regularizer):
    def __init__(self, weight):
        self.weight = weight
        super(Regularizer, self).__init__()

    def __call__(self, x):
        assert x.ndim == 2
        return self.weight * T.sqrt(T.sum(T.square(T.sum(T.abs_(x), axis=1)), axis=0))


def BalanceRegularizer(Regularizer):
    def __init__(self, weight):
        self.weight = weight
        super(Regularizer, self).__init__()

    def __call__(self, x):
        assert x.ndim == 2
        return self.weight * T.sum(T.log(T.mean(x, axis=1)), axis=0)
