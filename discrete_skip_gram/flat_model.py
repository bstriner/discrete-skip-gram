"""
Columnar analysis. Parameters are p(z|x). Each batch is a set of buckets.
"""

import csv
import os

import numpy as np
import theano
import theano.tensor as T
from .tensor_util import softmax_nd
from tqdm import tqdm

from .tensor_util import save_weights, load_latest_weights


class FlatModel(object):
    def __init__(self,
                 cooccurrence,
                 z_k,
                 opt,
                 pz_weight_regularizer=None,
                 pz_regularizer=None,
                 eps=1e-9,
                 scale=1e-2):
        cooccurrence = cooccurrence.astype(np.float32)
        self.cooccurrence = cooccurrence
        self.z_k = z_k
        x_k = cooccurrence.shape[0]
        self.x_k = x_k
        self.pz_weight_regularizer = pz_weight_regularizer
        self.pz_regularizer = pz_regularizer

        # cooccurrence matrix
        n = np.sum(cooccurrence, axis=None)
        _co = cooccurrence / n
        co_n = T.constant(_co, name="co_n")

        # parameters
        # P(z|x)
        initial_pz = np.random.uniform(-scale, scale, (x_k, z_k)).astype(np.float32)
        pz_weight = theano.shared(initial_pz, name="pz_weight")  # (x_k, z_k)
        params = [pz_weight]

        # p_z
        p_z = softmax_nd(pz_weight)  # (x_k, z_k)
        pzr = T.transpose(p_z, (1, 0))  # (z_k, x_k)

        # p(bucket)
        p_b = T.dot(pzr, co_n)  # (z_k, x_k)
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

        updates = opt.get_updates(params, {}, loss)

        train = theano.function([], [nll, reg_loss, loss], updates=updates)
        val = theano.function([], [nll, reg_loss, loss])
        encodings = theano.function([], p_z)

        self.train_fun = train
        self.val_fun = val
        self.encodings_fun = encodings
        self.z_fun = theano.function([], T.argmax(p_z, axis=1))  # (x_k,)

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

    def train(self, outputpath, epochs, batches):
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
                        nll, reg_loss, loss = self.train_fun()
                        it.desc = "Epoch {} NLL {:.4f} Reg Loss {:.4f} Loss {:.4f}".format(epoch,
                                                                                           np.asscalar(nll),
                                                                                           np.asscalar(reg_loss),
                                                                                           np.asscalar(loss))
                    w.writerow([epoch, nll, reg_loss, loss, self.calc_usage()])
                    f.flush()
                    enc = self.encodings_fun()  # (n, x_k)
                    np.save(os.path.join(outputpath, 'probabilities-{:08d}.npy'.format(epoch)), enc)
                    z = self.z_fun()  # (n,)
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), z)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
