import theano
import theano.tensor as T

def cumsum(x):
    y = T.transpose(x, (1, 0))
    z, _ = theano.scan(lambda _x, _pr: _x + _pr, sequences=[y], outputs_info=[T.zeros((x.shape[0],))])
    return T.transpose(z, (1, 0))


def cumprod(x):
    y = T.transpose(x, (1, 0))
    z, _ = theano.scan(lambda _x, _pr: _x * _pr, sequences=[y], outputs_info=[T.ones((x.shape[0],))])
    return T.transpose(z, (1, 0))