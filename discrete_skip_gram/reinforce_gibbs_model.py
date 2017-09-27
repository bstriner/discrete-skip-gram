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

from .tensor_util import save_weights, load_latest_weights
from .tensor_util import softmax_nd, tensor_one_hot


class ReinforceGibbsModel(object):
    def __init__(self,
                 cooccurrence,
                 z_k,
                 opt,
                 pz_weight_regularizer=None,
                 pz_regularizer=None,
                 eps=1e-8,
                 scale=1e-2,
                 beta=0.01,
                 batch_gibbs=True):
        srng = RandomStreams(123)
        cooccurrence = cooccurrence.astype(np.float32)
        self.cooccurrence = cooccurrence
        self.z_k = z_k
        self.opt = opt
        x_k = cooccurrence.shape[0]
        self.x_k = x_k
        self.pz_weight_regularizer = pz_weight_regularizer
        self.pz_regularizer = pz_regularizer
        self.batch_gibbs = batch_gibbs

        # cooccurrence matrix
        n = np.sum(cooccurrence, axis=None)
        _co = cooccurrence / n
        co = T.constant(_co, name="co")  # (x_k, x_k)
        _co_m = np.sum(_co, axis=1, keepdims=True)
        co_m = T.constant(_co_m, name="co_m")  # (x_k,1)
        _co_c = _co / (eps + _co_m)
        _co_h = np.sum(_co * -np.log(eps + _co_c), axis=1, keepdims=True)  # (x_k, 1)
        print "H(Y|X): {}".format(np.sum(_co_h))
        co_h = T.constant(_co_h, name="co_h")

        # parameters
        # P(z1=k,z2=k)
        tril = np.tril_indices(n=x_k, k=-1)
        initial_param = np.random.normal(loc=0, scale=scale, size=(tril[0].shape[0],)).astype(np.float32)
        param = K.variable(initial_param, name="param", dtype='float32')
        pz = T.zeros((x_k, x_k))
        pz = T.set_subtensor(pz[tril], param)
        pz += T.transpose(pz, (1, 0))  # symmetric
        pz = T.nnet.sigmoid(pz)  # (x_k, x_k) squash
        params = [param]

        # current sample
        initial_sample = np.random.random_integers(low=0, high=z_k - 1, size=(x_k,)).astype(np.int32)
        current_sample = K.variable(initial_sample, name="current_sample", dtype='int32')
        current_oh = tensor_one_hot(current_sample, k=z_k)  # (x_k, z_k)

        # probability of sample
        matches = T.eq(current_sample.dimshuffle((0, 'x')), current_sample.dimshuffle(('x', 0)))  # (x_k, x_k)
        p1 = T.nnet.sigmoid(param)
        p2 = matches[tril]
        lp = (p2 * T.log(eps + p1)) + ((1. - p2) * T.log(eps + 1 - p1))  # (tril,)
        sample_logp = T.sum(lp)

        # gibbs sampling
        if batch_gibbs:
            idx = T.ivector()
            pzidx = pz[idx, :]  # (n, x_k)
            current_masked = T.set_subtensor(current_oh[idx, current_sample[idx]], 0)  # (x_k, z_k)
            # todo: test this p calculation
            e_add = T.dot(T.log(eps + pzidx) - T.log(eps + 1 - pzidx), current_masked)  # (n, z_k)
            p_add = softmax_nd(e_add)
            cs = T.cumsum(p_add, axis=1)
            rnd = srng.uniform(low=0., high=1., size=(idx.shape[0],))
            bucket = T.sum(T.gt(rnd.dimshuffle((0, 'x')), cs), axis=1)  # (n,)
            bucket = T.clip(bucket, 0, z_k - 1)  # (n,)
            new_sample = T.set_subtensor(current_sample[idx], bucket)
            gibbs_updates = [(current_sample, new_sample)]
            self.gibbs_fun = theano.function([idx], [], updates=gibbs_updates)
        else:
            idx = srng.random_integers(low=0, high=x_k - 1)  # scalar
            pzidx = pz[idx, :]  # (x_k,)
            current_masked = T.set_subtensor(current_oh[idx, current_sample[idx]], 0)  # (x_k, z_k)
            # todo: test this p calculation
            e_add = T.dot(T.log(eps + pzidx) - T.log(eps + 1 - pzidx), current_masked)  # (Z_k,)
            p_add = softmax_nd(e_add)
            cs = T.cumsum(p_add)
            rnd = srng.uniform(low=0., high=1.)
            bucket = T.sum(T.gt(rnd, cs))
            bucket = T.clip(bucket, 0, z_k - 1)
            new_sample = T.set_subtensor(current_sample[idx], bucket)
            gibbs_updates = [(current_sample, new_sample)]
            self.gibbs_fun = theano.function([], [], updates=gibbs_updates)

        # loss of sample
        p_b = T.dot(T.transpose(current_oh, (1, 0)), co)  # (z_k, x_k)
        marg = T.sum(p_b, axis=1, keepdims=True)  # (z_k, 1)
        cond = p_b / (marg + eps)  # (z_k, x_k)
        current_nll = T.sum(p_b * -T.log(eps + cond), axis=None)  # scalar
        current_nll = theano.gradient.zero_grad(current_nll)

        avg_nll = K.variable(0., name='avg_nll', dtype='float32')
        new_avg = ((1. - beta) * avg_nll) + (beta * current_nll)
        avg_updates = [(avg_nll, new_avg)]

        # REINFORCE
        glp = T.grad(sample_logp, param)
        # todo: check sign
        sampled_grad = -(current_nll - avg_nll) * glp

        self.regularize = False

        assert isinstance(opt, keras.optimizers.Optimizer)

        def get_gradients(loss, params):
            assert len(params) == 1
            assert params[0] == param
            return [sampled_grad]

        opt.get_gradients = get_gradients
        updates = opt.get_updates(loss=current_nll, params=params)

        self.val_fun = theano.function([], current_nll)
        self.encodings_fun = theano.function([], current_sample)  # (x_k,)
        self.train_fun = theano.function([], current_nll, updates=updates + avg_updates)
        self.weights = params + opt.weights + [current_sample, avg_nll]

    def calc_utilization(self):
        z = self.encodings_fun()
        return len(set(list(z)))

    def train(self,
              outputpath,
              epochs,
              batches,
              steps,
              gibbs_batch_size=256):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
                w = csv.writer(f)
                w.writerow(['Epoch', 'Min NLL', 'Mean NLL', 'Utilization'])
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                    it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                    samples = []
                    for _ in it:
                        for _ in range(steps):
                            if self.batch_gibbs:
                                idx = np.random.choice(np.arange(self.x_k, dtype='int32'),
                                                       size=(gibbs_batch_size,),
                                                       replace=False)
                                self.gibbs_fun(idx)
                            else:
                                self.gibbs_fun()
                        nll = self.train_fun()
                        samples.append(nll)
                        vals = (epoch,
                                np.asscalar(np.min(samples)),
                                np.asscalar(np.mean(samples)),
                                np.asscalar(nll),
                                self.calc_utilization())
                        it.desc = ("Epoch {} Min NLL {:.4f} Mean NLL {:.4f}" +
                                   " Current NLL {:.4f} Utilization {}").format(*vals)
                    w.writerow([epoch,
                                np.asscalar(np.min(samples)),
                                np.asscalar(np.mean(samples)),
                                self.calc_utilization()])
                    f.flush()
                    enc = self.encodings_fun()  # (n, x_k)
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), enc)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
