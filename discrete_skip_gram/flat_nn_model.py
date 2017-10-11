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

from .tensor_util import save_weights, load_latest_weights, tensor_one_hot
from .tensor_util import softmax_nd


class FlatNNModel(object):
    def __init__(self,
                 cooccurrence,
                 z_k,
                 opt,
                 initializer,
                 pz_weight_regularizer=None,
                 pz_regularizer=None,
                 initial_pz=None,
                 initial_b=None,
                 eps=1e-8):
        cooccurrence = cooccurrence.astype(np.float32)
        self.cooccurrence = cooccurrence
        self.z_k = z_k
        self.opt = opt
        x_k = cooccurrence.shape[0]
        self.x_k = x_k
        self.pz_weight_regularizer = pz_weight_regularizer
        self.pz_regularizer = pz_regularizer

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
        if initial_pz is None:
            initial_pz = initializer((x_k, z_k))
        pz_weight = K.variable(initial_pz, name="pz_weight")  # (x_k, z_k)
        initial_w = initializer((z_k, x_k))
        w = K.variable(initial_w, name="w")
        if initial_b is None:
            initial_b = initializer((x_k,))
        b = K.variable(initial_b, name="b")
        params = [pz_weight, w, b]

        # loss
        p_z = softmax_nd(pz_weight)  # (x_k, z_k)
        bucketprobs = softmax_nd(w + b)  # (z_k, x_k)
        bucketnll = -T.log(eps + bucketprobs)  # (z_k, x_k)
        lossparts = T.dot(co, T.transpose(bucketnll, (1, 0)))  # (x_k, z_k)
        nll = T.sum(p_z * lossparts)

        # val loss
        enc = T.argmax(pz_weight, axis=1)
        oh = tensor_one_hot(enc, k=z_k)  # (x_k, z_k)
        p_b = T.dot(T.transpose(oh, (1, 0)), co)  # (z_k, x_k)
        marg = T.sum(p_b, axis=1, keepdims=True)  # (z_k, 1)
        cond = p_b / (marg + eps)  # (z_k, x_k)
        val_nll = T.sum(p_b * -T.log(eps + cond), axis=None)  # scalar

        # utilization
        utilization = T.sum(T.gt(T.sum(oh, axis=0), 0), axis=0)  # scalar

        reg_loss = T.constant(0.)
        self.regularize = False
        if pz_weight_regularizer:
            reg_loss += pz_weight_regularizer(pz_weight)
            self.regularize = True
        if pz_regularizer:
            reg_loss += pz_regularizer(p_z)
            self.regularize = True
        total_loss = nll + reg_loss

        self.val_fun = theano.function([], [nll, reg_loss, total_loss, val_nll, utilization])
        self.encodings_fun = theano.function([], enc)

        updates = opt.get_updates(params=params, loss=total_loss)
        self.train_fun = theano.function([], [nll, reg_loss, total_loss], updates=updates)
        self.weights = params + opt.weights

    def train_batch(self):
        return self.train_fun()

    def train(self,
              outputpath,
              epochs,
              batches):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        with open(os.path.join(outputpath, 'summary.txt'), 'w') as f:
            f.write("pz_weight_regularizer: {}\n".format(self.pz_weight_regularizer))
            f.write("pz_regularizer: {}\n".format(self.pz_regularizer))
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
                w = csv.writer(f)
                w.writerow(['Epoch', 'NLL', 'Reg loss', 'Loss', 'Val NLL', 'Utilization'])
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                    it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                    for _ in it:
                        nll, reg_loss, loss = self.train_batch()
                        it.desc = "Epoch {} NLL {:.4f} Reg Loss {:.4f} Loss {:.4f}".format(epoch,
                                                                                           np.asscalar(nll),
                                                                                           np.asscalar(reg_loss),
                                                                                           np.asscalar(loss))

                    nll, reg_loss, loss, val_nll, utilization = self.val_fun()
                    w.writerow([epoch, nll, reg_loss, loss, val_nll, utilization])
                    f.flush()
                    enc = self.encodings_fun()  # (n, x_k)
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), enc)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
