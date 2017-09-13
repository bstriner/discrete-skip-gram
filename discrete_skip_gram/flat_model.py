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

from .optimizers import Optimizer
from .tensor_util import save_weights, load_latest_weights
from .tensor_util import softmax_nd


class FlatModel(object):
    def __init__(self,
                 cooccurrence,
                 z_k,
                 opt,
                 pz_weight_regularizer=None,
                 pz_regularizer=None,
                 initial_pz=None,
                 eps=1e-9,
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
        _co_c = _co / _co_m
        _co_h = np.sum(_co * -np.log(eps + _co_c), axis=1, keepdims=True)  # (x_k, 1)
        print "COh: {}".format(np.sum(_co_h))
        co_h = T.constant(_co_h, name="co_h")

        # parameters
        # P(z|x)
        if initial_pz is None:
            initial_pz = np.random.normal(loc=0, scale=scale, size=(x_k, z_k)).astype(np.float32)
        pz_weight = K.variable(initial_pz, name="pz_weight")  # (x_k, z_k)
        params = [pz_weight]

        # p_z
        p_z = softmax_nd(T.reshape(pz_weight, (x_k, z_k)))  # (x_k, z_k)
        pzr = T.transpose(p_z, (1, 0))  # (z_k, x_k)

        # p(bucket)
        p_b = T.dot(pzr, co)  # (z_k, x_k)
        marg = T.sum(p_b, axis=1, keepdims=True)  # (z_k, 1)
        cond = p_b / (marg + eps)  # (z_k, x_k)
        nll = T.sum(p_b * -T.log(eps + cond), axis=None)  # scalar
        loss = nll

        reg_loss = T.constant(0.)
        if pz_weight_regularizer:
            reg_loss += pz_weight_regularizer(pz_weight)
        if pz_regularizer:
            reg_loss += pz_regularizer(p_z)
        loss += reg_loss

        self.val_fun = theano.function([], [nll, reg_loss, loss])
        self.encodings_fun = theano.function([], p_z)
        self.z_fun = theano.function([], T.argmax(p_z, axis=1))  # (x_k,)

        srng = RandomStreams(123)
        reset_opt = [(w, T.zeros_like(w)) for w in opt.weights]

        """
        # reset n rows
        reset_n = T.iscalar(name='reset_n')
        reset_x = T.arange(x_k)
        reset_idx = srng.choice(size=(reset_n,), a=reset_x, replace=False)
        
        reset_val = srng.normal(size=(reset_n, z_k), avg=0, std=scale)
        rval = T.set_subtensor(pz_weight[reset_idx, :], reset_val)
        reset_updates = [(pz_weight, rval)]
        self.reset_fun = theano.function([reset_n], [], updates=reset_updates + reset_opt)
        """

        # reset by softening

        soften = 1e0
        # reset_rnd = srng.normal(avg=0, std=scale, size=(x_k, z_k))
        # reset_updates = [(pz_weight, (pz_weight*soften) + reset_rnd)]
        m = T.mean(pz_weight, axis=1, keepdims=True)
        s = T.std(pz_weight, axis=1, keepdims=True)
        reset_updates = [(pz_weight, ((pz_weight - m) * soften / (eps + s)))]
        self.reset_fun = theano.function([], [], updates=reset_updates + reset_opt)

        if mode == 0:
            assert isinstance(opt, keras.optimizers.Optimizer)
            updates = opt.get_updates(params=params, loss=loss)
            self.train_fun = theano.function([], [nll, reg_loss, loss], updates=updates)
        elif mode == 1:
            # custom training
            nllc = -T.log(eps + cond)  # (z_k, x_k)
            # upper bound
            g2 = T.dot(co, T.transpose(nllc, (1, 0)))  # (x_k, z_k)
            # lower bound
            g3 = co_h
            # alpha
            # co_m (x_k, 1)
            # marg (z_k, 1)
            remain_p = 1 - p_z  # (x_k, z_k)
            remain_m = remain_p * co_m  # (x_k, z_k)
            alpha = remain_m / (remain_m + T.transpose(marg, (1, 0)))  # (x_k, z_k)

            # alpha = alpha**2

            gmerge = (alpha * g3) + ((1 - alpha) * g2)
            # todo: retry this line (w/w.o eps)
            # gmerge = T.log(eps + (alpha * T.exp(g3)) + ((1 - alpha) * T.exp(g2)))
            # gmerge = gmerge / (eps+co_m)
            # gmerge = gmerge * co_m * 1e2
            gmerge = theano.gradient.zero_grad(gmerge)

            s = T.sum(gmerge * p_z, axis=None)
            updates = opt.get_updates(params, {}, s)
            # g = T.grad(s, wrt=pz_weight)
            # lr = 1e-1
            # newp = pz_weight - (lr*g)
            # updates = [(pz_weight, newp)]
            self.train_fun = theano.function([], [nll, s, loss], updates=updates)
        elif mode == 2:
            assert isinstance(opt, Optimizer)
            idx = T.ivector(name='idx')  # [0-x_k] (n,)
            pzx = p_z[idx, :]  # (n, z_k)

            cox = co[idx, :]  # (n, x_k)
            h1 = (pzx.dimshuffle((0, 1, 'x'))) * (cox.dimshuffle((0, 'x', 1)))  # (n,z_k,x_k)
            p1 = (p_b.dimshuffle(('x', 0, 1))) - h1  # (n, z_k, x_k)
            m1 = T.sum(p1, axis=2, keepdims=True)
            c1 = p1 / (m1 + eps)
            e1 = T.sum(p1 * -T.log(eps+c1), axis=2)  # (n, z_k)

            remain = 1. - pzx  # (n, z_k)
            h2 = (remain.dimshuffle((0, 1, 'x'))) * (cox.dimshuffle((0, 'x', 1)))  # (n,z_k,x_k)
            p2 = (p_b.dimshuffle(('x', 0, 1))) + h2  # (n, z_k, x_k)
            m2 = T.sum(p2, axis=2, keepdims=True)
            c2 = p2 / (m2 + eps)
            e2 = T.sum(p2 * -T.log(eps+c2), axis=2)  # (n, z_k)

            d = e2 - e1  # (n, z_k)
            d = theano.gradient.zero_grad(d)
            subloss = T.sum(d * pzx, axis=None)  # scalar
            subloss += reg_loss
            opt.make_apply(params=params)
            self.train_fun = opt.make_train(inputs=[idx], outputs=[], loss=subloss)
        else:
            raise ValueError()

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

    def trainm2(self, batch_size=128):
        assert self.mode == 2
        i = 0
        while i < self.x_k:
            j = i + batch_size
            if j > self.x_k:
                j = self.x_k
            self.train_fun(np.arange(i, j).astype(np.int32))
            i = j
        self.opt.apply()
        return self.val_fun()

    def train_batch(self):
        if self.mode == 2:
            return self.trainm2()
        else:
            return self.train_fun()

    def train(self, outputpath, epochs,
              batches,
              watchdog=None,
              reset_n=50):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        with open(os.path.join(outputpath, 'summary.txt'), 'w') as f:
            f.write("pz_weight_regularizer: {}\n".format(self.pz_weight_regularizer))
            f.write("pz_regularizer: {}\n".format(self.pz_regularizer))
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
                w = csv.writer(f)
                w.writerow(['Epoch', 'NLL', 'Reg loss', 'Loss', 'Utilization'])
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                    it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                    for _ in it:
                        nll, reg_loss, loss = self.train_batch()
                        it.desc = "Epoch {} NLL {:.4f} Reg Loss {:.4f} Loss {:.4f}".format(epoch,
                                                                                           np.asscalar(nll),
                                                                                           np.asscalar(reg_loss),
                                                                                           np.asscalar(loss))
                        if watchdog and watchdog.check(loss):
                            self.reset_fun()

                    w.writerow([epoch, nll, reg_loss, loss, self.calc_usage()])
                    f.flush()
                    enc = self.encodings_fun()  # (n, x_k)
                    np.save(os.path.join(outputpath, 'probabilities-{:08d}.npy'.format(epoch)), enc)
                    z = self.z_fun()  # (n,)
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), z)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
