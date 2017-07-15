"""
Columnar analysis. Parameters are p(z|x). Each batch is a set of buckets.
"""

import csv
import os

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from discrete_skip_gram.skipgram.tensor_util import softmax_nd
from .optimizers import Optimizer
from .tensor_util import save_weights, load_latest_weights


class CategoricalColAccModel(object):
    def __init__(self, cooccurrence, z_k, opt,
                 type_np=np.float32,
                 type_t='float32',
                 scale=1e-1,
                 regularizer=None):
        assert isinstance(opt, Optimizer)
        cooccurrence = cooccurrence.astype(type_np)
        self.cooccurrence = cooccurrence
        self.type_np = type_np
        self.type_t = type_t
        self.z_k = z_k
        x_k = cooccurrence.shape[0]
        self.x_k = x_k
        self.opt = opt

        n = np.sum(cooccurrence, axis=None)
        # cooccurrence matrix
        co_n = T.constant(cooccurrence / n, name="co_n")

        # marginal probability
        _margin = np.sum(cooccurrence, axis=1) / n  # (x_k,)
        marg_p = T.constant(_margin)

        # conditional probability
        # _cond_p = cooccurrence / np.sum(cooccurrence, axis=1, keepdims=True)
        # cond_p = T.constant(_cond_p)  # (x_k,)

        # parameters
        # P(z|x)
        initial_weight = np.random.uniform(-scale, scale, (x_k, z_k)).astype(type_np)
        pz_weight = theano.shared(initial_weight, name="weight")  # (x_k, z_k)
        params = [pz_weight]

        # indices of columns
        idx = T.ivector()  # (n,) [0-z_k]

        # p_z
        p_z = softmax_nd(pz_weight)  # (x_k, z_k)
        p_zt = p_z[:, idx]  # (x_k, bn)

        # p(bucket)
        p_b = T.sum(p_zt * (marg_p.dimshuffle((0, 'x'))), axis=0)  # (bn,)

        h = T.sum(p_zt.dimshuffle((0, 1, 'x')) * (co_n.dimshuffle((0, 'x', 1))), axis=0)  # (bn, x_k)
        p_cond = h / T.sum(h, axis=1, keepdims=True)  # (bn, x_k)
        eps = T.constant(1e-9, dtype=type_t)
        nllpart = T.sum(p_cond * -T.log(eps + p_cond), axis=1)  # (bn,)
        nll = T.sum(p_b * nllpart)  # scalar
        # nll = T.sum(nllpart) / float(z_k) #* T.cast(idx.shape[0],type_t)
        loss = nll
        reg_loss = T.constant(0.)
        reg_weight = T.sum(p_b)
        if regularizer:
            for p in params:
                reg_loss += regularizer(p)
            reg_loss *= reg_weight
            loss += reg_loss

        opt.make_apply(params)
        self.train_fun = opt.make_train(inputs=[idx],
                                        outputs=[nll, reg_loss, loss],
                                        loss=loss)

        val = theano.function([idx], [nll, reg_loss, loss])

        encs = softmax_nd(pz_weight)
        # z = T.argmax(encs, axis=1)  #(x_k,)
        encodings = theano.function([], encs)
        self.encodings_fun = encodings
        self.val_fun = val
        self.all_weights = params + opt.weights

    def train_batch(self, idx, batch_size=32):
        nll = 0.
        loss = 0.
        reg_loss = 0.
        # np.random.shuffle(idx)
        n = idx.shape[0]
        batch_count = int(np.ceil(float(n) / float(batch_size)))
        for batch in range(batch_count):
            i1 = batch * batch_size
            i2 = (batch + 1) * batch_size
            if i2 > n:
                i2 = n
            b = idx[i1:i2]
            _nll, _reg_loss, _loss = self.train_fun(b)
            nll += _nll
            reg_loss += _reg_loss
            loss += _loss
        self.opt.apply()
        return nll, reg_loss, loss

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

    def train(self, outputpath, epochs, batches, batch_size):
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.all_weights)
        with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
            w = csv.writer(f)
            w.writerow(['Epoch', 'Reg loss', 'Loss', 'NLL'])
            f.flush()
            idx = np.arange(self.z_k).astype(np.int32)
            for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                for batch in it:
                    nll, reg_loss, loss = self.train_batch(idx=idx, batch_size=batch_size)
                    it.desc = "Epoch {} Reg Loss {:.4f} Loss {:.4f} NLL {:.4f}".format(epoch,
                                                                                       np.asscalar(reg_loss),
                                                                                       np.asscalar(loss),
                                                                                       np.asscalar(nll))
                w.writerow([epoch, reg_loss, loss, nll])
                f.flush()
                enc = self.encodings_fun()  # (n, x_k)
                np.save(os.path.join(outputpath, 'probabilities-{:08d}.npy'.format(epoch)), enc)
                # np.savetxt(os.path.join(outputpath, 'probabilities-{:08d}.txt'.format(epoch)), enc)
                z = np.argmax(enc, axis=1)  # (n,)
                np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), z)
                save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.all_weights)
