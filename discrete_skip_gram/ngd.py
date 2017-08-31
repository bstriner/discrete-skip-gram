import theano.gradient as G
import theano.tensor as T
from keras.optimizers import Optimizer
from .diff import diff2

class NGD(Optimizer):
    def __init__(self, lr=1e-3):
        self.lr = T.constant(lr, name='lr')
        super(NGD, self).__init__()

    def get_updates(self, params, constraints, loss):
        for p in params:
            assert p.ndim == 1
        eps = 1e-9
        g = self.get_gradients(loss=loss, params=params)
        #g2 = [T.diag(G.hessian(loss, p)) for p in params]
        g2 = [diff2(cost=loss, wrt=p) for p in params]
        updates = [(p, p - self.lr * (a / (eps+a2))) for p, a, a2 in zip(params, g, g2)]
        return updates
