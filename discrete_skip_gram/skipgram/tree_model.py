import csv
import os

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm
from discrete_skip_gram.skipgram.tensor_util import softmax_nd
from .tensor_util import save_weights, load_latest_weights
from .util import array_string


class TreeModel(object):
    def __init__(self,
                 cooccurrence,
                 z_depth,
                 z_k,
                 opt,
                 schedule,
                 pz_regularizer=None,
                 pz_weight_regularizer=None,
                 eps=1e-9,
                 scale=1e-2,
                 use_shared=True):
        cooccurrence = cooccurrence.astype(np.float32)
        cooccurrence = cooccurrence / np.sum(cooccurrence, axis=None)
        if use_shared:
            co = theano.shared(cooccurrence, name='cooccurrence')
        else:
            co = T.constant(cooccurrence, dtype='float32', name='cooccurrence')
        self.cooccurrence = cooccurrence
        self.z_depth = z_depth
        self.z_k = z_k
        self.x_k = cooccurrence.shape[0]
        self.opt = opt
        self.schedule = schedule
        self.pz_regularizer = pz_regularizer
        self.pz_weight_regularizer = pz_weight_regularizer
        assert schedule.shape[0] == z_depth
        assert schedule.ndim == 1
        x_k = cooccurrence.shape[0]
        schedule = T.constant(schedule.astype(np.float32), dtype='float32', name="schedule")  # (z_depth,)

        # parameterization
        buckets = int(z_k ** z_depth)
        initial_weight = np.random.uniform(-scale, scale, (x_k, buckets)).astype(np.float32)
        pz_weight = theano.shared(initial_weight, name="pz_weight")  # (x_k, z_k)
        self.params = [pz_weight]
        pz0 = softmax_nd(pz_weight)  # (x_k, z_k)
        # calculate p(z|x)
        pzs = []
        for depth in range(0, z_depth - 1):
            b0 = int(z_k ** (depth + 1))
            h = T.reshape(pz0, (x_k, b0, -1))  # (x_k, b0, -1)
            pzt = T.sum(h, axis=2)  # (x_k, b0)
            pzs.append(pzt)
        pzs.append(pz0)
        self.pzs = pzs

        # calculate nlls
        nll_array = []
        for depth in range(z_depth):
            pz = pzs[depth]  # (x_k, b0)
            p = T.dot(co, pz)  # (x_k, b0)

            marg = T.sum(p, axis=0, keepdims=True)  # (1, b0)
            cond = p / (marg + eps)  # (x_k, b0)
            nll = T.sum(p * - T.log(cond + eps), axis=None)  # scalar
            nll_array.append(nll)
        nlls = T.stack(nll_array)  # (z_depth,)
        loss = T.sum(schedule * nlls, axis=0)  # scalar

        # regularization
        reg_loss = T.constant(0.)
        if pz_weight_regularizer:
            reg_loss += pz_weight_regularizer(pz_weight)
        if pz_regularizer:
            pz_loss = []
            for pz in pzs:
                pz_loss.append(pz_regularizer(pz))
            reg_loss += T.sum(T.stack(pz_loss) * schedule)

        # training
        loss += reg_loss
        updates = opt.get_updates(self.params, {}, loss)

        # encoding
        z = T.argmax(pz0, axis=1)  # (x_k,) [int 0-buckets]
        zt = z
        encodings = []
        for depth in range(z_depth):
            c = int(z_k ** (z_depth - depth - 1))
            enc = T.ge(zt, c)
            zt -= (c * enc)
            encodings.append(enc)
        encodings = T.stack(encodings, axis=1)  # (x_k, z_depth)

        # Theano functions
        self.train_fun = theano.function([], [nlls, reg_loss, loss], updates=updates)
        self.val_fun = theano.function([], [nlls, reg_loss, loss])
        self.encodings_fun = theano.function([], encodings)
        self.z_fun = theano.function([], z)

        self.weights = self.params + opt.weights

    def calc_usage(self):
        z = self.z_fun()
        s = set(z[i] for i in range(z.shape[0]))
        assert z.shape[0] == self.x_k
        return len(s)

    def train_batch(self):
        return self.train_fun()

    def train(self, outputpath, epochs, batches):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
                w = csv.writer(f)
                w.writerow(
                    ['Epoch', 'Reg Loss', 'Loss', 'Utilization'] + ['NLL {}'.format(i) for i in range(self.z_depth)])
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                    it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                    for _ in it:
                        nll, reg_loss, loss = self.train_batch()
                        it.desc = "Epoch {} Reg Loss {:.4f} Loss {:.4f} NLL [{}]".format(epoch,
                                                                                         np.asscalar(reg_loss),
                                                                                         np.asscalar(loss),
                                                                                         array_string(nll))
                    w.writerow([epoch, reg_loss, loss, self.calc_usage()] +
                               [np.asscalar(nll[i]) for i in range(self.z_depth)])
                    f.flush()
                    enc = self.encodings_fun()  # (n, z_depth) [int 0-z_k]
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), enc)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
        return self.val_fun()

