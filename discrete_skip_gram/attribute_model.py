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


class AttributeModel(object):
    def __init__(self,
                 cooccurrence,
                 zks,
                 a_k,
                 opt,
                 initial_pz=None,
                 eps=1e-8,
                 scale=1e-2,
                 mode=0):
        cooccurrence = cooccurrence.astype(np.float32)
        self.cooccurrence = cooccurrence
        self.z_ks = z_ks
        self.a_k = a_k
        self.opt = opt
        x_k = cooccurrence.shape[0]
        self.x_k = x_k
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
        params = []
        pzs = []
        nlls = []
        for zk in zks:
            initial_pz = np.random.normal(loc=0, scale=scale, size=(x_k, zk)).astype(np.float32)
            pz_weight = K.variable(initial_pz, name="pz_weight")  # (x_k, z_k)
            params.append(pz_weight)
            pz = T.nnet.softmax(pz_weight) # (x_k, z_k)
            pzs.append(pz)

            pzr = T.transpose(pz, (1, 0))  # (z_k, x_k)
            pb = T.dot(pzr, co) # (z_k, x_k)
            marg = T.sum(pb, axis=1, keepdims=True)  # (z_k, 1)
            cond = pb / (marg + eps)  # (z_k, x_k)
            nll = T.sum(pb * -T.log(eps + cond), axis=None)  # scalar
            nlls.append(nll)
        nlls = T.stack(nlls)
        loss = T.sum(nlls)

        l = len(zks)
        reg_loss = T.constant(0., dtype='float32')
        rweight = 1e-9
        for i in range(l-1):
            z1 = pzs[i] # p(z1|x)
            zh = z1 * co_m # p(z1,x)
            for j in range(i+1, l):
                z2 = pzs[j] # p(z2|x)
                h = T.dot(zh, T.transpose(z2, (1,0))) # p(x1,x2) (z1, z2)
                reg_loss += rweight * T.sum(-T.log(eps+h))

        total_loss = loss + reg_loss
        encodings = [T.argmax(pz, axis=1) for pz in pzs]

        self.val_fun = theano.function([], [nlls, reg_loss, loss])
        self.encodings_fun = theano.function([], pzs)
        self.z_fun = theano.function([], encodings)  # (x_k,)

        updates = opt.get_updates(params=params, loss=total_loss)
        self.train_fun = theano.function([], [nlls, reg_loss, total_loss], updates=updates)
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

    def trainm2(self, batch_size=64):
        assert self.mode == 2
        i = 0
        while i < self.x_k:
            j = i + batch_size
            if j > self.x_k:
                j = self.x_k
            self.train_fun(np.arange(i, j).astype(np.int32))
            i = j
        if self.regularize:
            ret = self.train_reg_fun()
            self.opt.apply()
            return ret
        else:
            self.opt.apply()
            return self.val_fun()

    def trainm3(self, batch_size=64):
        assert self.mode == 3
        idx = np.random.choice(a=np.arange(self.x_k, dtype=np.int32), size=batch_size, replace=False)
        return self.train_fun(idx)

    def train_batch(self):
        if self.mode == 2:
            return self.trainm2()
        elif self.mode == 3:
            return self.trainm3()
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
