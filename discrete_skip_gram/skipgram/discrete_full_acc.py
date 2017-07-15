import csv
import os

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from discrete_skip_gram.skipgram.tensor_util import softmax_nd
from .tensor_util import save_weights, load_latest_weights
from .util import array_string


class DiscreteFullAccModel(object):
    def __init__(self,
                 cooccurrence,
                 z_depth,
                 z_k,
                 opt,
                 schedule,
                 type_np=np.float32,
                 type_t='float32',
                 regularizer=None):
        cooccurrence = cooccurrence.astype(type_np)
        self.cooccurrence = cooccurrence
        self.type_np = type_np
        self.type_t = type_t
        self.z_depth = z_depth
        scale = 1e-2
        x_k = cooccurrence.shape[0]
        schedule = T.constant(schedule.astype(type_np), dtype=type_t, name="schedule")  # (z_depth,)

        # cooccurrence
        cooccurrence_n = T.constant((cooccurrence/np.sum(cooccurrence,axis=None)).astype(type_np))

        # marginal probability
        n = np.sum(cooccurrence, axis=None)
        _margin = np.sum(cooccurrence, axis=1) / n  # (x_k,)
        marg_p = T.constant(_margin, dtype=type_t)
        log_marg_p = T.constant(np.log(_margin)-np.max(np.log(_margin)), dtype=type_t) # (x_k,)

        # conditional probability
        _cond_p = cooccurrence / np.sum(cooccurrence, axis=1, keepdims=True)
        cond_p = T.constant(_cond_p, dtype=type_t)  # (x_k,)

        # parameters
        # p(z|x) weights
        pz_weights = []
        for depth in range(z_depth):
            buckets = int(z_k ** depth)
            initial_weight = np.random.uniform(-scale, scale, (x_k, buckets, z_k)).astype(type_np)
            pz_weight = theano.shared(initial_weight, name="pz_{}".format(depth))  # (x_k, buckets, z_k)
            pz_weights.append(pz_weight)

        params = pz_weights

        # indices
        idx = T.imatrix()  # (n, z_depth)
        #n = idx.shape[0]

        # calculate p(z|x)
        p0 = T.ones((x_k, 1, 1), dtype=type_t)  # (n, b0, z_k)
        pzs = []
        for depth in range(z_depth):
            p = softmax_nd(pz_weights[depth])  # (x_k, b1, z_k)
            h = T.reshape(p0, (p0.shape[0], p0.shape[1] * p0.shape[2]))  # (x_k, b1)
            p1 = (h.dimshuffle((0, 1, 'x'))) * p  # (x_k, b1, z_k)
            p0 = p1
            pzs.append(p1)

        pzts = []
        for depth in range(z_depth):


        # loss calculation
        nlls = []
        for depth in range(z_depth):
            nll = self.calc_depth(pzs[depth], py_weights[depth]+log_marg_p, cond_pt)  # (n,)
            nlls.append(nll)
        nlls = T.stack(nlls, axis=1)  # (n, z_depth)
        wnlls = T.sum(nlls * (marg_pt.dimshuffle((0, 'x'))), axis=0)  # (z_depth,)
        loss = T.sum(schedule * wnlls, axis=0)  # scalar
        reg_loss = 0.
        if regularizer:
            for p in params:
                reg_loss += regularizer(p)
            reg_loss *= T.sum(marg_pt) # scale to size of batch
            loss += reg_loss
        updates = opt.get_updates(params, {}, loss)
        train = theano.function([idx], [wnlls, reg_loss, loss], updates=updates)

        # Discrete encoding
        e0 = T.zeros((x_k,), dtype='int32')  # (x_k,)
        encs = []
        for depth in range(z_depth):
            p = softmax_nd(pz_weights[depth])  # (x_k, buckets, z_k)
            enc = T.argmax(p[T.arange(p.shape[0]), e0, :], axis=1)  # (x_k,) [int 0-z_k]
            assert enc.ndim == 1
            e1 = (e0 * z_k) + enc  # (x_k,) [int 0-b1] todo: double-check order
            e0 = e1
            encs.append(enc)
        encoding = T.stack(encs, axis=1)  # (x_k, z_depth)
        encodings = theano.function([], encoding)
        self.train_fun = train
        self.encodings_fun = encodings
        self.all_weights = params + opt.weights

    def calc_depth(self, pz, py_weight, cond_pt):
        # pz: (n, buckets, z_k)
        # py_weight: (buckets, z_k, x_k)
        # cond_pt: (n, x_k)
        py = softmax_nd(py_weight)  # (buckets, z_k, x_k)
        eps = 1e-9
        nll = -T.log(eps + py)  # (buckets, z_k, x_k)
        loss1 = (cond_pt.dimshuffle((0, 'x', 'x', 1))) * (nll.dimshuffle(('x', 0, 1, 2)))  # (n, buckets, z_k, x_k)
        loss2 = T.sum(loss1, axis=3)  # (n, buckets, z_k)
        loss3 = T.sum(loss2 * pz, axis=[1, 2])  # (n,)
        assert loss3.ndim == 1
        return loss3

    def train_batch(self, idx, batch_size=32):
        nll = np.zeros((self.z_depth,), dtype=self.type_np)
        reg_loss = 0.
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
            _nll, _reg_loss, _loss = self.train_fun(b)
            nll += _nll
            reg_loss += _reg_loss
            loss += _loss
        return nll, reg_loss, loss

    def train(self, outputpath, epochs, batches, batch_size):
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.all_weights)
        with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
            w = csv.writer(f)
            w.writerow(['Epoch', 'Reg Loss', 'Loss'] + ['NLL {}'.format(i) for i in range(self.z_depth)])
            f.flush()
            idx = np.arange(self.cooccurrence.shape[0]).astype(np.int32)
            for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                for batch in it:
                    nll, reg_loss, loss = self.train_batch(idx=idx, batch_size=batch_size)
                    it.desc = "Epoch {} Reg Loss {:.4f} Loss {:.4f} NLL [{}]".format(epoch,
                                                                                     np.asscalar(reg_loss),
                                                                                     np.asscalar(loss),
                                                                                     array_string(nll))
                w.writerow([epoch, reg_loss, loss] + [np.asscalar(nll[i]) for i in range(self.z_depth)])
                f.flush()
                enc = self.encodings_fun()  # (n, z_depth) [int 0-z_k]
                np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), enc)
                save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.all_weights)
