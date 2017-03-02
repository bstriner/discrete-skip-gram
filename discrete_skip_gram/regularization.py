import theano.tensor as T


def l1l2(l1, l2):
    def regularizer(p):
        return l1 * T.sum(T.abs_(p), axis=None) + l2 * T.sqrt(T.sum(T.square(p), axis=None))

    return regularizer
