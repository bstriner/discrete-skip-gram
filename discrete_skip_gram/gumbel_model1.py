import csv
import os

import numpy as np
import theano
import theano.tensor as T
from keras import backend as K
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from .tensor_util import save_weights, load_latest_weights
from .tensor_util import softmax_nd, tensor_one_hot


class GumbelModel1(object):
    def __init__(self,
                 cooccurrence,
                 z_k,
                 opt,
                 initializer,
                 initial_pz_weight=None,
                 initial_b=None,
                 pz_regularizer=None,
                 tao0=5.,
                 tao_min=0.25,
                 tao_decay=1e-6,
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
        pz = softmax_nd(pz_weight)

        srng = RandomStreams(123)
        rnd = srng.uniform(low=0., high=1., dtype='float32', size=(x_k, z_k))
        gumbel = -T.log(eps + T.nnet.relu(-T.log(eps + rnd)))

        iteration = K.variable(0, dtype='int32')
        temp = T.max(T.stack((tao_min, tao0 / (1. + (tao_decay * iteration)))))

        z = softmax_nd((pz_weight + gumbel) / (eps + temp))
        # z = pz
        w = K.variable(initializer((z_k, x_k)))
        if initial_b is None:
            initial_b = initializer((x_k,))
        b = K.variable(initial_b)
        y = softmax_nd(T.dot(z, w) + b)

        self.params = [pz_weight, w, b]

        nll_loss = -T.sum(co * T.log(eps + y), axis=None)
        reg_loss = T.constant(0.)
        if pz_regularizer:
            reg_loss = pz_regularizer(pz)
        total_loss = nll_loss + reg_loss

        decay_updates = [(iteration, iteration + 1)]

        encoding = T.argmax(pz, axis=1)
        one_hot_encoding = tensor_one_hot(encoding, z_k)  # (x_k, z_k)

        pb = T.dot(T.transpose(one_hot_encoding, (1, 0)), co)
        m = T.sum(pb, axis=1, keepdims=True)
        c = pb / (m + eps)
        validation_nll = -T.sum(pb * T.log(eps + c), axis=None)

        utilization = T.sum(T.gt(T.sum(one_hot_encoding, axis=0), 0), axis=0)
        updates = opt.get_updates(loss=total_loss, params=self.params)

        self.val_fun = theano.function([], validation_nll)
        self.encodings_fun = theano.function([], encoding)
        self.train_fun = theano.function([], [reg_loss, nll_loss, utilization, temp],
                                         updates=updates + decay_updates)
        self.weights = self.params + opt.weights + [iteration]

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
                            'Mean Loss',
                            'Min Loss',
                            'Mean Utilization',
                            'Min Utilization',
                            'Max Utilization',
                            'Temperature',
                            'Validation NLL'])
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                    it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                    reg_losses = []
                    losses = []
                    utilizations = []
                    temp = None
                    for _ in it:
                        reg_loss, loss, utilization, temp = self.train_batch()
                        reg_losses.append(reg_loss)
                        losses.append(loss)
                        utilizations.append(utilization)
                        it.desc = ("Epoch {}: " +
                                   "Reg Loss {:.4f} " +
                                   "Mean Loss {:.4f} " +
                                   "Current Loss {:.4f} " +
                                   "Current Utilization {} " +
                                   "Current Temperature {:.4f}").format(epoch,
                                                                        np.asscalar(reg_loss),
                                                                        np.asscalar(np.mean(losses)),
                                                                        np.asscalar(np.min(losses)),
                                                                        np.asscalar(utilization),
                                                                        np.asscalar(temp))
                    validation_nll = self.val_fun()
                    w.writerow([epoch,
                                np.asscalar(np.mean(reg_losses)),
                                np.asscalar(np.mean(losses)),
                                np.asscalar(np.min(losses)),
                                np.asscalar(np.mean(utilizations)),
                                np.asscalar(np.min(utilizations)),
                                np.asscalar(np.max(utilizations)),
                                np.asscalar(temp),
                                np.asscalar(validation_nll)])
                    f.flush()
                    enc = self.encodings_fun()  # (n, x_k)
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), enc)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
