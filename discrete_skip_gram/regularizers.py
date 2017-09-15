import keras.regularizers
import theano.tensor as T


class Regularizer(keras.regularizers.Regularizer):
    def __init__(self, weight):
        self.weight = weight
        super(Regularizer, self).__init__()

    def __str__(self):
        return "{}(weight={})".format(self.__class__.__name__, self.weight)


class L2Mean(Regularizer):
    def __init__(self, weight, axis):
        self.axis = axis
        super(L2Mean, self).__init__(weight)

    def __call__(self, x):
        return self.weight * T.sum(T.square(T.mean(x, axis=self.axis)), axis=None)


class L2(Regularizer):
    def __init__(self, weight):
        super(L2, self).__init__(weight)

    def __call__(self, x):
        return self.weight * T.sum(T.square(x), axis=None)


class ExclusiveLasso(Regularizer):
    def __init__(self, weight):
        super(ExclusiveLasso, self).__init__(weight)

    def __call__(self, x):
        assert x.ndim == 2
        return self.weight * T.sum(T.square(T.sum(T.abs_(x), axis=0)), axis=0)


class ExclusiveLassoSqrt(Regularizer):
    def __init__(self, weight):
        super(ExclusiveLassoSqrt, self).__init__(weight)

    def __call__(self, x):
        assert x.ndim == 2
        return self.weight * T.sqrt(T.sum(T.square(T.sum(T.abs_(x), axis=0)), axis=0))


class BalanceRegularizer(Regularizer):
    def __init__(self, weight):
        super(BalanceRegularizer, self).__init__(weight)

    def __call__(self, x):
        assert x.ndim == 2
        return -self.weight * T.sum(T.log(T.mean(x, axis=0)), axis=0)


class BalanceWeightedRegularizer(Regularizer):
    def __init__(self, weight, marginal):
        self.marginal = marginal
        super(BalanceWeightedRegularizer, self).__init__(weight)

    def __call__(self, x):
        assert x.ndim == 2
        p = T.sum(x * (self.marginal.dimshuffle((0, 'x'))), axis=0)
        return -self.weight * T.sum(T.log(p), axis=0)
