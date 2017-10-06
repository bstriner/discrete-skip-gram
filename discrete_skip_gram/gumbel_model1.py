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
                 decay=1e-6,
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
        rnd = srng.uniform(low=0. + eps, high=1. - eps, dtype='float32', size=(x_k, z_k))
        gumbel = -T.log(-T.log(rnd))

        iteration = K.variable(0, dtype='int32')
        temp = 1. / (1. + (decay * iteration))

        z = softmax_nd((T.log(pz) + gumbel) / temp)

        w = K.variable(initializer((z_k, x_k)))
        b = K.variable(initializer((x_k,)))
        y = softmax_nd(T.dot(z, w) + b)

        self.params = [pz_weight, w, b]

        loss = -T.sum(co * T.log(eps + y), axis=None)

        decay_updates = [(iteration, iteration + 1)]

        encoding = T.argmax(pz, axis=1)
        one_hot_encoding = tensor_one_hot(encoding, z_k)

        utilization = T.sum(T.gt(T.sum(one_hot_encoding, axis=0), 0), axis=0)
        updates = opt.get_updates(loss=loss, params=self.params)

        self.val_fun = theano.function([], [loss])
        self.encodings_fun = theano.function([], encoding)
        self.train_fun = theano.function([], [loss, utilization, temp], updates=updates + decay_updates)
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
                            'Mean Loss',
                            'Current Loss',
                            'Mean Utilization',
                            'Min Utilization',
                            'Max Utilization',
                            'Temperature'])
                f.flush()
                for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                    it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                    losses = []
                    utilizations = []
                    temp = None
                    for _ in it:
                        loss, utilization, temp = self.train_batch()
                        losses.append(loss)
                        utilizations.append(utilization)
                        it.desc = ("Epoch {}: " +
                                   "Mean Loss {:.4f} " +
                                   "Current Loss {:.4f} " +
                                   "Current Utilization {} " +
                                   "Current Temperature {:.4f}").format(epoch,
                                                                         np.asscalar(np.mean(losses)),
                                                                         np.asscalar(np.min(losses)),
                                                                         np.asscalar(utilization),
                                                                         np.asscalar(temp))
                    w.writerow([epoch,
                                np.asscalar(np.mean(losses)),
                                np.asscalar(np.min(losses)),
                                np.asscalar(np.mean(utilizations)),
                                np.asscalar(np.min(utilizations)),
                                np.asscalar(np.max(utilizations)),
                                np.asscalar(temp)])
                    f.flush()
                    enc = self.encodings_fun()  # (n, x_k)
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), enc)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
