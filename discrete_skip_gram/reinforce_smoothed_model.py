import csv
import os

import numpy as np
import theano
import theano.tensor as T
from keras import backend as K
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from .tensor_util import sample_from_distribution
from .tensor_util import save_weights, load_latest_weights
from .tensor_util import softmax_nd, tensor_one_hot


class ReinforceSmoothedModel(object):
    def __init__(self,
                 cooccurrence,
                 z_k,
                 opt,
                 initializer,
                 tau0=2.,
                 initial_pz_weight=None,
                 initial_b=None,
                 pz_regularizer=None,
                 beta=0.02,
                 eps=1e-9):
        cooccurrence = cooccurrence.astype(np.float32)
        self.cooccurrence = cooccurrence
        self.z_k = z_k
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

        if initial_pz_weight is None:
            initial_pz_weight = initializer((x_k, z_k))
        pz_weight = K.variable(initial_pz_weight)
        initial_w = initializer((z_k, x_k))
        w = K.variable(initial_w, name="w")  # (z_k, x_k)
        if initial_b is None:
            initial_b = initializer((x_k,))
        b = K.variable(initial_b, name="b")
        yw = softmax_nd(w + b)  # (z_k, x_k)

        srng = RandomStreams(123)
        p = softmax_nd(pz_weight)
        if tau0 > 0:
            q = softmax_nd(pz_weight / tau0)
            sample = sample_from_distribution(q, srng=srng)
            pt = p[T.arange(p.shape[0]), sample]
            qt = p[T.arange(q.shape[0]), sample]
            factor = theano.gradient.zero_grad(pt / qt)
        else:
            sample = sample_from_distribution(p, srng=srng)
            pt = p[T.arange(p.shape[0]), sample]
            factor = 1.

        ysamp = yw[sample, :]
        nllpart = -T.sum(co * T.log(eps + ysamp), axis=1)  # (x,)
        nll_loss = T.sum(nllpart)

        initial_nll = theano.function([], nllpart)()
        avg_nll = theano.shared(np.float32(initial_nll), name='avg_nll')
        new_avg = ((1. - beta) * avg_nll) + (beta * nllpart)
        avg_updates = [(avg_nll, new_avg)]
        reinforce = theano.gradient.zero_grad((nllpart - avg_nll) * factor)
        rloss = T.sum(reinforce * T.log(pt), axis=0)
        nloss = T.sum(nllpart * factor, axis=0)

        # updates
        self.params = [pz_weight, w, b]
        reg_loss = T.constant(0.)
        if pz_regularizer:
            reg_loss = pz_regularizer(p)
        total_loss = rloss + nloss + reg_loss
        updates = opt.get_updates(total_loss, self.params)

        encoding = T.argmax(pz_weight, axis=1)
        one_hot_encoding = tensor_one_hot(encoding, z_k)  # (x_k, z_k)

        pb = T.dot(T.transpose(one_hot_encoding, (1, 0)), co)
        m = T.sum(pb, axis=1, keepdims=True)
        c = pb / (m + eps)
        validation_nll = -T.sum(pb * T.log(eps + c), axis=None)

        utilization = T.sum(T.gt(T.sum(one_hot_encoding, axis=0), 0), axis=0)

        self.val_fun = theano.function([], [validation_nll, utilization])
        self.encodings_fun = theano.function([], encoding)
        self.train_fun = theano.function([], [reg_loss, nll_loss, total_loss, sample],
                                         updates=updates + avg_updates)
        self.weights = self.params + opt.weights + [avg_nll]

    def train_batch(self):
        return self.train_fun()

    def train(self, outputpath, epochs,
              batches):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
                w = csv.writer(f)
                w.writerow(['Epoch',
                            'Mean Reg Loss',
                            'Mean NLL Loss',
                            'Mean Total Loss',
                            'Validation NLL',
                            'Utilization'])
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                    it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                    data = [[] for _ in range(3)]
                    minenc = None
                    minnll = None
                    for _ in it:
                        reg_loss, nll_loss, loss, enc = self.train_batch()
                        if (minnll is None) or (nll_loss < minnll):
                            minnll = nll_loss
                            minenc = enc
                        for i, d in enumerate((reg_loss, nll_loss, loss)):
                            data[i].append(d)
                        it.desc = ("Epoch {}: " +
                                   "Reg Loss {:.4f} " +
                                   "NLL Loss {:.4f} " +
                                   "Mean NLL {:.4f} " +
                                   "Min NLL {:.4f} " +
                                   "Current Loss {:.4f} " +
                                   "Mean Loss {:.4f} " +
                                   "Min Loss {:.4f}").format(epoch,
                                                             np.asscalar(reg_loss),
                                                             np.asscalar(nll_loss),
                                                             np.asscalar(np.mean(data[1])),
                                                             np.asscalar(np.min(data[1])),
                                                             np.asscalar(loss),
                                                             np.asscalar(np.mean(data[2])),
                                                             np.asscalar(np.min(data[2])))
                    val_nll, utilization = self.val_fun()
                    w.writerow([epoch,
                                np.asscalar(np.mean(data[0])),
                                np.asscalar(np.mean(data[1])),
                                np.asscalar(np.mean(data[2])),
                                np.asscalar(val_nll),
                                np.asscalar(utilization)])
                    f.flush()
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), minenc)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
