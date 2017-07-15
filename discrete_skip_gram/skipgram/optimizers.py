import numpy as np
import theano
import theano.tensor as T

import keras.backend as K


class Optimizer(object):
    def __init__(self):
        self.train_start_fun = None
        self.apply_fun = None
        self.accs = None
        self.weights_np = None
        self.weights = []
        self.params = None

    def zeros_like(self):
        return [theano.shared(np.zeros(w.shape, dtype=w.dtype)) for w in self.weights_np]

    def make_apply(self, params):
        assert self.params is None
        self.params = params
        self.weights_np = [K.get_value(p) for p in params]
        self.accs = self.zeros_like()
        apply_updates = self.make_apply_updates()
        clear_updates = [(acc, np.zeros(w.shape, dtype=w.dtype)) for acc, w in zip(self.accs, self.weights_np)]
        updates = apply_updates + clear_updates
        self.apply_fun = theano.function([], [], updates=updates)

    def make_apply_updates(self):
        raise NotImplementedError("make_apply_updates not implemented")

    def apply(self):
        self.apply_fun()

    def make_train(self, inputs, outputs, loss, disconnected_inputs='raise'):
        grads = [T.grad(loss, p, disconnected_inputs=disconnected_inputs) for p in self.params]
        grads1 = [acc + grad for acc, grad in zip(self.accs, grads)]
        grad_updates = [(acc, grad1) for acc, grad1 in zip(self.accs, grads1)]
        train_fun = theano.function(inputs, outputs=outputs, updates=grad_updates)
        return train_fun


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

    def make_apply_updates(self):
        ms = self.zeros_like()
        vs = self.zeros_like()
        self.weights += ms + vs
        ms1 = []
        vs1 = []
        params1 = []
        for p, g, m, v in zip(self.params, self.accs, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - self.lr * m_t / (K.sqrt(v_t) + self.epsilon)

            ms1.append(m_t)
            vs1.append(v_t)
            params1.append(p_t)

        m_updates = [(a, b) for a, b in zip(ms, ms1)]
        v_updates = [(a, b) for a, b in zip(vs, vs1)]
        w_updates = [(a, b) for a, b in zip(self.params, params1)]
        updates = m_updates + v_updates + w_updates
        return updates
