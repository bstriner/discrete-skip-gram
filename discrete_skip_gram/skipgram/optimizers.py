import numpy as np
import theano
import theano.tensor as T

import keras.backend as K


class Optimizer(object):
    def __init__(self):
        self.train_fun = None
        self.train_start_fun = None
        self.apply_fun = None
        self.accs = None
        self.weights = []

    def make_accumulators(self, weights):
        ws = [K.get_value(w) for w in weights]
        self.accs = [theano.shared(np.zeros(w.shape, dtype=w.dtype)) for w in ws]

    def make_apply_updates(self, weights, ws, grads):
        raise NotImplementedError("make_apply_updates not implemented")

    def train(self, inputs):
        return self.train_fun(inputs)

    def apply(self):
        self.apply_fun()

    #    def start(self):
    #        self.train_start_fun([])

    def make_functions(self, inputs, outputs, loss, weights):
        ws = [K.get_value(w) for w in weights]
        accs = [theano.shared(np.zeros(w.shape, dtype=w.dtype)) for w in ws]
        # self.weights += accs

        grads = [T.grad(loss, w) for w in weights]
        grads1 = [acc + grad for acc, grad in zip(accs, grads)]
        grad_updates = [(acc, grad1) for acc, grad1 in zip(accs, grads1)]

        clear_updates = [(acc, np.zeros(w.shape, dtype=w.dtype)) for acc, w in zip(accs, ws)]

        apply_updates = self.make_apply_updates(weights=weights, ws=ws, grads=accs)
        apply_fun_updates = apply_updates + clear_updates

        self.train_fun = theano.function(inputs, outputs=outputs, updates=grad_updates)
        self.train_start_fun = theano.function([], [], updates=clear_updates)
        self.apply_fun = theano.function([], [], updates=apply_fun_updates)


class AdamOptimizer(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8):
        # self.lr = K.variable(lr, name='lr')
        self.lr = T.constant(lr, name='lr')
        assert self.lr.ndim == 0
        self.beta_1 = K.variable(beta_1, name='beta_1')
        self.beta_2 = K.variable(beta_2, name='beta_2')
        self.epsilon = epsilon
        super(AdamOptimizer, self).__init__()



    def make_apply_updates(self, weights, ws, grads):
        ms = [theano.shared(np.zeros(w.shape, dtype=w.dtype)) for w in ws]
        vs = [theano.shared(np.zeros(w.shape, dtype=w.dtype)) for w in ws]
        self.weights += ms + vs
        ms1 = []
        vs1 = []
        weights1 = []
        for p, g, m, v in zip(weights, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - self.lr * m_t / (K.sqrt(v_t) + self.epsilon)

            ms1.append(m_t)
            vs1.append(v_t)
            weights1.append(p_t)

        m_updates = [(a, b) for a, b in zip(ms, ms1)]
        v_updates = [(a, b) for a, b in zip(vs, vs1)]
        w_updates = [(a, b) for a, b in zip(weights, weights1)]
        updates = m_updates + v_updates + w_updates
        return updates
