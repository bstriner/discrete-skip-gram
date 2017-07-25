import csv
import os

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from .tensor_util import save_weights, load_latest_weights
from .tree_parameterization import ParameterizationFull
from .util import array_string, generate_sequences, generate_batches


class DiscreteFullAccModel(object):
    def __init__(self,
                 cooccurrence,
                 z_depth,
                 z_k,
                 opt,
                 schedule,
                 param_class=ParameterizationFull,
                 type_np=np.float32,
                 type_t='float32',
                 regularizer=None):
        cooccurrence = cooccurrence.astype(type_np)
        self.cooccurrence = cooccurrence
        self.type_np = type_np
        self.type_t = type_t
        self.z_k = z_k
        self.z_depth = z_depth
        self.eps = T.constant(1e-9, dtype=type_t)
        self.opt = opt
        x_k = cooccurrence.shape[0]
        schedule = T.constant(schedule.astype(type_np), dtype=type_t, name="schedule")  # (z_depth,)

        # self.modes = [0 if 2 ** i > 8 else 1 for i in range(z_depth)]

        # cooccurrence
        cooccurrence_n = T.constant((cooccurrence / np.sum(cooccurrence, axis=None)).astype(type_np))

        # parameters
        # p(z|x) weights
        parameterization = param_class(z_k=z_k, x_k=x_k, z_depth=z_depth, type_np=type_np, type_t=type_t)
        params = parameterization.params

        opt.make_apply(params)

        # regularization
        self.regularizer_fun = None
        if regularizer or parameterization.loss:
            reg_loss = T.constant(0., dtype=type_t)
            if regularizer:
                for p in params:
                    reg_loss += regularizer(p)
            if parameterization.loss:
                reg_loss += parameterization.loss
            self.regularizer_fun = opt.make_train([], reg_loss, reg_loss)

        # indices
        idx = T.imatrix()  # (n, z_depth)

        self.train_funs = []
        for depth in range(self.z_depth):
            pz = parameterization.pzs[depth]
            weight = schedule[depth]
            # f0
            nll0 = self.calc_depth_full(pz=pz, cooccurrence_n=cooccurrence_n)
            loss0 = nll0 * weight
            fun0 = opt.make_train(inputs=[], outputs=[nll0, loss0], loss=loss0, disconnected_inputs='ignore')
            # f1
            nll1 = self.calc_depth_part(pz=pz, cooccurrence_n=cooccurrence_n, idx=idx)
            loss1 = nll1 * weight
            fun1 = opt.make_train(inputs=[idx], outputs=[nll1, loss1], loss=loss1, disconnected_inputs='ignore')
            self.train_funs.append([fun0, fun1])

        self.encodings_fun=None
        if parameterization.encoding:
            self.encodings_fun = theano.function([], parameterization.encoding)
        self.probs_fun = theano.function([], parameterization.pzs)
        self.all_weights = params + opt.weights

    def calc_depth_full(self, pz, cooccurrence_n):
        """
        If buckets are < ~8
        :param pz:
        :param co_n:
        :return:
        """
        # pz: (x_k, z_k)
        # co_n: (x_k, x_k)
        h = (pz.dimshuffle((0, 1, 'x'))) * (cooccurrence_n.dimshuffle((0, 'x', 1)))  # (x_k, z_k, x_k)
        p = T.sum(h, axis=0)  # (z_k, x_k)
        marg = T.sum(p, axis=1, keepdims=True)  # (z_k,1)
        cond = p / marg  # (z_k, x_k)
        nll = T.sum(cond * -T.log(self.eps + cond), axis=1)  # (z_k,)
        loss = T.sum(nll * (marg.dimshuffle((0,))))
        return loss

    def calc_depth_part(self, pz, cooccurrence_n, idx):
        """
        If buckets are > ~8
        :param pz:
        :param co_n:
        :return:
        """
        # pz: (x_k, z_k)
        # co_n: (x_k, x_k)
        # idx: (bn, depth)
        mask = T.power(self.z_k, T.arange(idx.shape[1], dtype='int32'))  # (depth,)
        buckets = T.sum((mask.dimshuffle(('x', 0))) * idx, axis=1)  # (bn,) [int32]
        pzt = pz[:, buckets]  # (x_k, bn)

        h = (pzt.dimshuffle((0, 1, 'x'))) * (cooccurrence_n.dimshuffle((0, 'x', 1)))  # (x_k, bn, x_k)
        p = T.sum(h, axis=0)  # (bn, x_k)
        marg = T.sum(p, axis=1, keepdims=True)  # (bn,1)
        cond = p / marg  # (bn, x_k)
        nll = T.sum(cond * -T.log(self.eps + cond), axis=1)  # (bn,)
        loss = T.sum(nll * (marg.dimshuffle((0,))))
        return loss

    def train_depth_full(self, depth):
        fun = self.train_funs[depth][0]
        return fun()

    def train_depth_part(self, depth, batches):
        nll = 0.
        loss = 0.
        fun = self.train_funs[depth][1]
        for b in batches:
            _nll, _loss = fun(b)
            nll += _nll
            loss += _loss
        return np.asscalar(nll), np.asscalar(loss)

    def train_batch(self, modes, batch_data):
        nll = np.zeros((self.z_depth,), dtype=self.type_np)
        loss = 0.
        for depth, (mode, data) in enumerate(zip(modes, batch_data)):
            if mode == 0:
                _nll, _loss = self.train_depth_full(depth)
                nll[depth] = _nll
                loss += _loss
            elif mode == 1:
                _nll, _loss = self.train_depth_part(depth, batch_data[depth])
                nll[depth] = _nll
                loss += _loss
            else:
                raise ValueError("invalid mode")
        reg_loss = 0.
        if self.regularizer_fun:
            reg_loss = self.regularizer_fun()
        self.opt.apply()
        loss += reg_loss
        return nll, reg_loss, loss

    def train(self, outputpath, epochs, batches, batch_size):
        modes = [0 if 2 ** (i + 1) <= batch_size else 1 for i in range(self.z_depth)]
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.all_weights)
        train_data = [generate_batches(generate_sequences(i, self.z_k), batch_size=batch_size)
                      for i in range(self.z_depth)]
        with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
            w = csv.writer(f)
            w.writerow(['Epoch', 'Reg Loss', 'Loss'] + ['NLL {}'.format(i) for i in range(self.z_depth)])
            f.flush()
            for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                for batch in it:
                    nll, reg_loss, loss = self.train_batch(modes, train_data)
                    it.desc = "Epoch {} Reg Loss {:.4f} Loss {:.4f} NLL [{}]".format(epoch,
                                                                                     np.asscalar(reg_loss),
                                                                                     np.asscalar(loss),
                                                                                     array_string(nll))
                w.writerow([epoch, reg_loss, loss] + [np.asscalar(nll[i]) for i in range(self.z_depth)])
                f.flush()
                if self.encodings_fun:
                    enc = self.encodings_fun()  # (n, z_depth) [int 0-z_k]
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), enc)
                    pzs = self.probs_fun()
                    np.savez(os.path.join(outputpath, 'pz-{:08d}.npz'.format(epoch)), *pzs)
                save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.all_weights)
