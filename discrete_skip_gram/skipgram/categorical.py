import csv
import os

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from discrete_skip_gram.skipgram.tensor_util import softmax_nd


class CategoricalModel(object):
    def __init__(self, cooccurrence, z_k, opt, regularizer=None):
        self.cooccurrence = cooccurrence
        scale = 1e-1
        x_k = cooccurrence.shape[0]

        # marginal probability
        n = np.sum(cooccurrence, axis=None)
        _margin = np.sum(cooccurrence, axis=1) / n  # (x_k,)
        marg_p = T.constant(_margin)

        # conditional probability
        _cond_p = cooccurrence / np.sum(cooccurrence, axis=1, keepdims=True)
        cond_p = T.constant(_cond_p)  # (x_k,)

        # parameters
        initial_weight = np.random.uniform(-scale, scale, (x_k, z_k)).astype(np.float32)
        pz_weight = theano.shared(initial_weight, name="weight")  # (x_k, z_k)
        initial_py = np.random.uniform(-scale, scale, (z_k, x_k)).astype(np.float32)  # (z_k, x_k)
        py_weight = theano.shared(initial_py, name='py')  # (z_k, x_k)
        params = [pz_weight, py_weight]

        # indices
        idx = T.ivector()  # (n,)

        # p_z
        p_z = softmax_nd(pz_weight[idx, :])  # (n, z_k)

        # p_y
        p_y = softmax_nd(py_weight)  # (z_k, x_k)
        eps = 1e-8
        nll_y = -T.log(p_y + eps)  # (z_k, x_k)

        co_pt = cond_p[idx, :]  # (n, x_k)
        h = (nll_y.dimshuffle(('x', 0, 1))) * (co_pt.dimshuffle((0, 'x', 1)))  # (n, z_k, x_k)
        losspart = T.sum(h, axis=2)  # (n, z_k)

        # loss
        lossn = T.sum(losspart * p_z, axis=1)  # (n,)
        marg_pt = marg_p[idx]  # (n,)
        nll = T.sum(lossn * marg_pt, axis=0)
        loss = nll
        if regularizer:
            for p in params:
                loss += regularizer(p)
        updates = opt.get_updates(params, {}, loss)
        train = theano.function([idx], [nll, loss], updates=updates)

        encs = softmax_nd(pz_weight)
        encodings = theano.function([], encs)
        self.train_fun = train
        self.encodings_fun = encodings

    def train_batch(self, idx, batch_size=32):
        nll = 0.
        loss = 0.
        np.random.shuffle(idx)
        n = idx.shape[0]
        batch_count = int(np.ceil(float(n) / float(batch_size)))
        for batch in range(batch_count):
            i1 = batch * batch_size
            i2 = (batch + 1) * batch_size
            if i2 > n:
                i2 = n
            b = idx[i1:i2]
            _nll, _loss = self.train_fun(b)
            nll += _nll
            loss += _loss
        return nll, loss

    def train(self, outputpath, epochs, batches, batch_size):
        with open(os.path.join(outputpath, 'history.csv'), 'wb') as f:
            w = csv.writer(f)
            w.writerow(['Epoch', 'Loss', 'NLL'])
            f.flush()
            idx = np.arange(self.cooccurrence.shape[0]).astype(np.int32)
            for epoch in tqdm(range(epochs), desc="Training"):
                it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                for batch in it:
                    nll, loss = self.train_batch(idx=idx, batch_size=batch_size)
                    it.desc = "Epoch {} Loss {:.4f} NLL {:.4f}".format(epoch, np.asscalar(loss), np.asscalar(nll))
                w.writerow([epoch, loss, nll])
                f.flush()
                enc = self.encodings_fun()  # (n, x_k)
                np.save(os.path.join(outputpath, 'probabilities-{:08d}.npy'.format(epoch)), enc)
                z = np.argmax(enc, axis=1)  # (n,)
                np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), z)
