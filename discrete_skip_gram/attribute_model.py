"""
Columnar analysis. Parameters are p(z|x). Each batch is a set of buckets.
"""

import csv
import os

import numpy as np
import theano
import theano.tensor as T
from keras import backend as K
from tqdm import tqdm

from .tensor_util import save_weights, load_latest_weights
from .util import array_string


class AttributeModel(object):
    def __init__(self,
                 cooccurrence,
                 zks,
                 opt,
                 eps=1e-9,
                 scale=1e-2,
                 reg_weight=1e-9):
        cooccurrence = cooccurrence.astype(np.float32)
        self.cooccurrence = cooccurrence
        self.zks = zks
        self.opt = opt
        x_k = cooccurrence.shape[0]
        self.x_k = x_k

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
        loss = 0.
        for zk in zks:
            initial_pz = np.random.normal(loc=0, scale=scale, size=(x_k, zk)).astype(np.float32)
            pz_weight = K.variable(initial_pz, name="pz_weight")  # (x_k, z_k)
            params.append(pz_weight)
            pz = T.nnet.softmax(pz_weight)  # (x_k, z_k)
            pzs.append(pz)

            pzr = T.transpose(pz, (1, 0))  # (z_k, x_k)
            pb = T.dot(pzr, co)  # (z_k, x_k)
            marg = T.sum(pb, axis=1, keepdims=True)  # (z_k, 1)
            cond = pb / (marg + eps)  # (z_k, x_k)
            nll = T.sum(pb * -T.log(eps + cond), axis=None)  # scalar
            loss += nll
            nlls.append(nll)
        nlls = T.stack(nlls)

        l = len(zks)
        reg_loss = 0.
        if reg_weight > 0:
            for i in range(l - 1):
                z1 = pzs[i]  # p(z1|x)
                zh = z1 * co_m  # p(z1,x)
                for j in range(i + 1, l):
                    z2 = pzs[j]  # p(z2|x)
                    h = T.dot(zh, T.transpose(z2, (1, 0)))  # p(x1,x2) (z1, z2)
                    reg_loss += reg_weight * T.sum(-T.log(eps + h), axis=None)

        total_loss = loss + reg_loss
        encodings = [T.argmax(pz, axis=1) for pz in pzs]

        self.encodings_fun = theano.function([], pzs)
        self.z_fun = theano.function([], encodings)  # (x_k,)

        updates = opt.get_updates(params=params, loss=total_loss)
        self.train_fun = theano.function([], [nlls, reg_loss, total_loss], updates=updates)
        self.weights = params + opt.weights

        # validation
        b = 0
        for i in range(l):
            if i < l - 1:
                b += encodings[i] * np.prod(zks[(1 + i):])
            else:
                b += encodings[i]
        h = T.zeros((x_k, np.prod(zks)), dtype='float32')
        h = T.set_subtensor(h[T.arange(x_k), b], 1.)
        v_pb = T.dot(T.transpose(h, (1, 0)), co)  # (b, x_k)
        v_m = T.sum(v_pb, axis=1, keepdims=True)
        v_c = v_pb / (eps + v_m)
        v_nll = -T.sum(v_pb * T.log(eps + v_c), axis=None)
        self.val_fun = theano.function([], [nlls, reg_loss, loss, v_nll])

    def calc_usage(self):
        zs = self.z_fun()
        usages = [len(set(list(z))) for z in zs]
        return usages

    def train(self,
              outputpath,
              epochs,
              batches):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
                w = csv.writer(f)
                w.writerow(['Epoch', 'Reg loss', 'Loss', 'Val Loss'] +
                           ['NLL {}'.format(i) for i, _ in enumerate(self.zks)] +
                           ['Utilization {}'.format(i) for i, _ in enumerate(self.zks)])
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                    it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                    for _ in it:
                        nlls, reg_loss, loss = self.train_fun()
                        it.desc = "Epoch {} NLLs [{}] Reg Loss {:.4f} Loss {:.4f}".format(epoch,
                                                                                          array_string(nlls),
                                                                                          np.asscalar(reg_loss),
                                                                                          np.asscalar(loss))
                    nlls, reg_loss, loss, v_nll = self.val_fun()
                    w.writerow([epoch, reg_loss, loss, v_nll] +
                               [np.asscalar(nll) for nll in nlls] +
                               self.calc_usage())
                    f.flush()
                    # enc = self.encodings_fun()  # (n, x_k)
                    # np.save(os.path.join(outputpath, 'probabilities-{:08d}.npy'.format(epoch)), enc)
                    z = self.z_fun()  # (n,)
                    np.savez(os.path.join(outputpath, 'encodings-{:08d}.npz'.format(epoch)), z)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
