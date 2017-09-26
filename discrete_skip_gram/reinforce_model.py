"""
Columnar analysis. Parameters are p(z|x). Each batch is a set of buckets.
"""

import csv
import os

import keras
import numpy as np
import theano
import theano.tensor as T
from keras import backend as K
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from .tensor_util import save_weights, load_latest_weights
from .tensor_util import softmax_nd


class ReinforceModel(object):
    def __init__(self,
                 cooccurrence,
                 z_k,
                 opt,
                 pz_weight_regularizer=None,
                 pz_regularizer=None,
                 eps=1e-8,
                 scale=1e-2,
                 mode=0):
        cooccurrence = cooccurrence.astype(np.float32)
        self.cooccurrence = cooccurrence
        self.z_k = z_k
        self.opt = opt
        x_k = cooccurrence.shape[0]
        self.x_k = x_k
        self.pz_weight_regularizer = pz_weight_regularizer
        self.pz_regularizer = pz_regularizer
        self.mode = mode

        # cooccurrence matrix
        n = np.sum(cooccurrence, axis=None)
        _co = cooccurrence / n
        co = T.constant(_co, name="co")  # (x_k, x_k)
        _co_m = np.sum(_co, axis=1, keepdims=True)
        co_m = T.constant(_co_m, name="co_m")  # (x_k,1)
        _co_c = _co / (eps + _co_m)
        _co_h = np.sum(_co * -np.log(eps + _co_c), axis=1, keepdims=True)  # (x_k, 1)
        print "COh: {}".format(np.sum(_co_h))
        co_h = T.constant(_co_h, name="co_h")

        # parameters
        # P(z|x)
        initial_pz = np.random.normal(loc=0, scale=scale, size=(x_k, z_k)).astype(np.float32)
        pz_weight = K.variable(initial_pz, name="pz_weight")  # (x_k, z_k)
        params = [pz_weight]

        # p_z
        p_z = softmax_nd(T.reshape(pz_weight, (x_k, z_k)))  # (x_k, z_k)

        # p(bucket)
        p_b = T.dot(T.transpose(p_z, (1, 0)), co)  # (z_k, x_k)
        marg = T.sum(p_b, axis=1, keepdims=True)  # (z_k, 1)
        cond = p_b / (marg + eps)  # (z_k, x_k)
        base_nll = T.sum(p_b * -T.log(eps + cond), axis=None)  # scalar

        # sampled
        srng = RandomStreams(123)
        rng = srng.uniform(low=0, high=1, size=(x_k,))
        cs = T.cumsum(p_z, axis=1)
        sel = T.sum(T.gt(rng.dimshuffle((0, 'x')), cs), axis=1)  # (x_k,)

        b = T.zeros((z_k, x_k))
        b = T.set_subtensor(b[sel, T.arange(x_k)], 1)
        pb = T.dot(b, co)
        m = T.sum(pb, axis=1, keepdims=True)
        c = pb / (m + eps)
        sampled_nll = T.sum(pb * -T.log(eps + c), axis=None)  # scalar
        sampled_nll = theano.gradient.zero_grad(sampled_nll)
        sampled_lp = T.sum(T.log(p_z[T.arange(x_k), sel]))  # scalar
        glp = T.grad(sampled_lp, pz_weight)
        sampled_grad = sampled_nll * glp
        grad_tot = sampled_grad

        reg_loss = T.constant(0.)
        self.regularize = False
        if pz_weight_regularizer:
            reg_loss += pz_weight_regularizer(pz_weight)
            self.regularize = True
        if pz_regularizer:
            reg_loss += pz_regularizer(p_z)
            self.regularize = True
        if self.regularize:
            reg_grad = T.grad(reg_loss, pz_weight)
            grad_tot += reg_grad

        tot_loss = base_nll + reg_loss

        assert isinstance(opt, keras.optimizers.Optimizer)

        def get_gradients(loss, params):
            assert len(params)==1
            assert params[0] == pz_weight
            return [grad_tot]

        opt.get_gradients = get_gradients

        updates = opt.get_updates(loss=base_nll, params=params)

        self.val_fun = theano.function([], [base_nll, reg_loss, tot_loss])
        self.encodings_fun = theano.function([], p_z)
        self.z_fun = theano.function([], T.argmax(p_z, axis=1))  # (x_k,)
        self.train_fun = theano.function([], [base_nll, reg_loss, sampled_nll], updates=updates)
        self.weights = params + opt.weights

    def calc_usage(self):
        z = self.z_fun()
        s = set(z[i] for i in range(z.shape[0]))
        return len(s)

    def validate(self, batch_size=32):
        nll = 0.
        loss = 0.
        reg_loss = 0.
        idx = np.arange(self.z_k, dtype=np.int32)
        n = idx.shape[0]
        batch_count = int(np.ceil(float(n) / float(batch_size)))
        for batch in range(batch_count):
            i1 = batch * batch_size
            i2 = (batch + 1) * batch_size
            if i2 > n:
                i2 = n
            b = idx[i1:i2]
            _nll, _reg_loss, _loss = self.val_fun(b)
            nll += _nll
            reg_loss += _reg_loss
            loss += _loss
        return nll, reg_loss, loss

    def train_batch(self):
        return self.train_fun()

    def train(self, outputpath, epochs,
              batches):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
                w = csv.writer(f)
                w.writerow(['Epoch', 'NLL', 'Reg loss', 'Total Loss', 'Mean Sampled Loss', 'Utilization'])
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                    it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                    samples = []
                    for _ in it:
                        nll, reg_loss, sampled_loss = self.train_batch()
                        it.desc = "Epoch {} NLL {:.4f} Reg Loss {:.4f} Sampled Loss {:.4f}".format(epoch,
                                                                                                   np.asscalar(nll),
                                                                                                   np.asscalar(
                                                                                                       reg_loss),
                                                                                                   np.asscalar(
                                                                                                       sampled_loss))
                        samples.append(sampled_loss)
                    sampled = np.mean(samples)
                    nll, reg_loss, tot_loss = self.val_fun()
                    w.writerow([epoch, nll, reg_loss, tot_loss, sampled, self.calc_usage()])
                    f.flush()
                    enc = self.encodings_fun()  # (n, x_k)
                    np.save(os.path.join(outputpath, 'probabilities-{:08d}.npy'.format(epoch)), enc)
                    z = self.z_fun()  # (n,)
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), z)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
