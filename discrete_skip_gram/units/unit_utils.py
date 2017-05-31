import theano.tensor as T


def batch_normalization(x):
    return T.nnet.bn.batch_normalization_test(inputs=x,
                                              gamma=T.ones(x.shape[1:], dtype='float32'),
                                              beta=T.zeros(x.shape[1:], dtype='float32'),
                                              mean=T.mean(x, axis=0),
                                              var=T.var(x, axis=0),
                                              axes=(0,))
