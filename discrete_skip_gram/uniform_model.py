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


class UniformModel(object):
    def __init__(self,
                 cooccurrence,
                 z_k,
                 opt,
                 initializer,
                 initial_pz_weight=None,
                 initial_b=None,
                 pz_regularizer=None,
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
        initial_w = initializer((z_k, x_k))
        w = K.variable(initial_w, name="w")  # (z_k, x_k)
        if initial_b is None:
            initial_b = initializer((x_k,))
        b = K.variable(initial_b, name="b")
        yw = softmax_nd(w+b)  # (z_k, x_k)
        srng = RandomStreams(123)
        zsamp = srng.random_integers(size=(x_k,), low=0, high=z_k - 1)

        yt = yw[zsamp, :]  # (x_k, x_k)
        lt = -T.sum(co * T.log(eps + yt), axis=1)  # (x_k,)
        pt = pz[T.arange(pz.shape[0]), zsamp]
        assert lt.ndim == 1
        assert pt.ndim == 1
        nll_loss = T.sum(pt * lt, axis=None) * z_k

        self.params = [pz_weight, w, b]
        reg_loss = T.constant(0.)
        if pz_regularizer:
            reg_loss = pz_regularizer(pz)
        total_loss = nll_loss + reg_loss

        encoding = T.argmax(pz_weight, axis=1)
        one_hot_encoding = tensor_one_hot(encoding, z_k)  # (x_k, z_k)

        pb = T.dot(T.transpose(one_hot_encoding, (1, 0)), co)
        m = T.sum(pb, axis=1, keepdims=True)
        c = pb / (m + eps)
        validation_nll = -T.sum(pb * T.log(eps + c), axis=None)

        utilization = T.sum(T.gt(T.sum(one_hot_encoding, axis=0), 0), axis=0)
        updates = opt.get_updates(loss=total_loss, params=self.params)

        self.val_fun = theano.function([], [validation_nll, utilization])
        self.encodings_fun = theano.function([], encoding)
        self.train_fun = theano.function([], [reg_loss, nll_loss, total_loss],
                                         updates=updates)
        self.weights = self.params + opt.weights

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
                    for _ in it:
                        reg_loss, nll_loss, loss = self.train_batch()
                        for i, d in enumerate((reg_loss, nll_loss, loss)):
                            data[i].append(d)
                        it.desc = ("Epoch {}: " +
                                   "Reg Loss {:.4f} " +
                                   "NLL Loss {:.4f} " +
                                   "Mean NLL {:.4f} " +
                                   "Current Loss {:.4f} " +
                                   "Mean Loss {:.4f} " +
                                   "Min Loss {:.4f}").format(epoch,
                                                             np.asscalar(reg_loss),
                                                             np.asscalar(nll_loss),
                                                             np.asscalar(np.mean(data[1])),
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
                    enc = self.encodings_fun()  # (n, x_k)
                    np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), enc)
                    save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
