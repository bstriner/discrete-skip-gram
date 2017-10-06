import csv
import os

import numpy as np
import theano
import theano.tensor as T
from keras import backend as K
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from .tensor_util import save_weights, load_latest_weights
from .tensor_util import softmax_nd


class ReinforceFactoredModel(object):
    def __init__(self,
                 cooccurrence,
                 z_k,
                 opt,
                 initializer,
                 initial_pz_weight=None,
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

        # parameters
        if initial_pz_weight is None:
            initial_pz_weight = initializer((x_k, z_k))
        else:
            print("Using stored weights")
        pz_weight = K.variable(initial_pz_weight)
        params = [pz_weight]

        # p(z|x)
        pz = softmax_nd(pz_weight)

        # sample
        sample_mode = 0
        srng = RandomStreams(123)
        if sample_mode == 0:
            cs = T.cumsum(pz, axis=1)
            rnd = srng.uniform(low=0., high=1., dtype='float32', size=(x_k,))
            encoding = T.sum(T.gt(rnd.dimshuffle((0, 'x')), cs), axis=1)
            encoding = T.clip(encoding, 0, z_k - 1)
        elif sample_mode == 1:
            rnd = srng.uniform(low=0. + eps, high=1. - eps, dtype='float32', size=(x_k, z_k))
            h = T.log(pz) - T.log(-T.log(rnd))
            encoding = T.argmax(h, axis=1)
        else:
            raise ValueError()

        # log p
        pzt = pz[T.arange(x_k), encoding]  # (x_k,)
        logpzt = T.log(pzt)  # (x_k,)

        # nll
        b = T.zeros((z_k, x_k))
        b = T.set_subtensor(b[encoding, T.arange(x_k)], 1)  # one-hot encoding (z_k, x_k)
        pb = T.dot(b, co)  # (z_k, x_k)
        m = T.sum(pb, axis=1, keepdims=True)  # (z_k, 1)
        c = pb / (m + eps)  # (z_k, x_k)
        nll1 = T.dot(co, T.transpose(-T.log(eps + c), (1, 0)))  # (x_k, x_k) * (x_k, z_k) = (x_k, z_k)
        nll2 = nll1 * T.transpose(b, (1, 0))  # (x_k, z_k)
        nllpart = T.sum(nll2, axis=1)  # (x_k,)
        nlltot = T.sum(nllpart)  # scalar

        # moving average
        avg_nll = K.variable(np.zeros((x_k,)), dtype='float32')
        new_avg = (beta * nllpart) + ((1. - beta) * avg_nll)
        avg_updates = [(avg_nll, new_avg)]

        # initialize moving average
        theano.function([], [], updates=[(avg_nll, nllpart)])()

        # todo: check sign!
        r = theano.gradient.zero_grad(nllpart - avg_nll)
        loss = T.sum(r * logpzt)  # scalar

        reg_loss = T.constant(0.)
        if pz_regularizer:
            reg_loss = pz_regularizer(pz)

        total_loss = loss + reg_loss

        utilization = T.sum(T.gt(T.sum(b, axis=1), 0), axis=0)

        updates = opt.get_updates(loss=total_loss, params=params)

        self.val_fun = theano.function([], [nlltot, total_loss, utilization])
        self.encodings_fun = theano.function([], encoding)
        self.train_fun = theano.function([], [nlltot, reg_loss, total_loss, utilization, encoding],
                                         updates=updates + avg_updates)
        self.weights = params + opt.weights + [avg_nll]

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
                            'Mean NLL',
                            'Min NLL',
                            'Mean Utilization',
                            'Min Utilization',
                            'Max Utilization',
                            'Regularization Loss',
                            'Loss'])
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                    it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                    nlls = []
                    reg_losses = []
                    losses = []
                    utilizations = []
                    min_nll = None
                    min_enc = None
                    for _ in it:
                        nll, reg_loss, loss, utilization, enc = self.train_batch()
                        nlls.append(nll)
                        reg_losses.append(reg_loss)
                        losses.append(loss)
                        utilizations.append(utilization)
                        if (min_nll is None) or (nll < min_nll):
                            min_enc = enc
                            min_nll = nll
                        it.desc = ("Epoch {}: " +
                                   "Mean NLL {:.4f} " +
                                   "Min NLL {:.4f} " +
                                   "Current NLL {:.4f} " +
                                   "Current Utilization {} " +
                                   "Mean Regularization Loss {:.4f} " +
                                   "Mean Loss {:.4f} " +
                                   "Current Loss {:.4f}").format(epoch,
                                                                 np.asscalar(np.mean(nlls)),
                                                                 np.asscalar(np.min(nlls)),
                                                                 np.asscalar(nll),
                                                                 np.asscalar(utilization),
                                                                 np.asscalar(np.mean(reg_losses)),
                                                                 np.asscalar(np.mean(losses)),
                                                                 np.asscalar(loss))
                    w.writerow([epoch,
                                np.asscalar(np.mean(nlls)),
                                np.asscalar(np.min(nlls)),
                                np.asscalar(np.mean(utilizations)),
                                np.asscalar(np.min(utilizations)),
                                np.asscalar(np.max(utilizations)),
                                np.asscalar(np.mean(reg_losses)),
                                np.asscalar(np.mean(losses))])
                    f.flush()
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), min_enc)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
