import csv
import os

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from keras.optimizers import Optimizer
from .tensor_util import save_weights, load_latest_weights
from keras.optimizers import Adam
from .util import make_path

class BaselineModel(object):
    def __init__(self,
                 cooccurrence,
                 z_units,
                 opt,
                 regularizer=None,
                 scale=1e-1,
                 eps=1e-9):
        x_k = cooccurrence.shape[0]
        assert isinstance(opt, Optimizer)
        self.opt = opt

        # parameters
        initial_embedding = np.random.uniform(-scale, scale, (x_k, z_units)).astype(np.float32)
        initial_weight = np.random.uniform(-scale, scale, (z_units, x_k)).astype(np.float32)
        initial_bias = np.random.uniform(-scale, scale, (x_k,)).astype(np.float32)
        self.embedding = theano.shared(initial_embedding, name="embedding")
        self.weight = theano.shared(initial_weight, name="weight")
        self.bias = theano.shared(initial_bias, name="bias")

        # normalize
        _co = cooccurrence.astype(np.float32)
        _co = _co / np.sum(_co, axis=None)
        co = T.constant(_co)

        # p
        h = T.dot(self.embedding, self.weight) + self.bias
        p = T.nnet.softmax(h)  # (x_k, x_k)
        nll = T.sum(co * -T.log(eps + p), axis=None)  # scalar
        loss = nll
        if regularizer:
            reg = regularizer(self.weight) + regularizer(self.embedding)
            loss += reg

        self.params = [self.embedding, self.weight, self.bias]
        updates = self.opt.get_updates(self.params, {}, loss)
        self.train_fun = theano.function([], [nll, loss], updates=updates)
        self.encodings_fun = theano.function([], self.embedding)
        self.val_fun = theano.function([], [nll, loss])
        self.weights = self.params + self.opt.weights

    def train(self, outputpath, epochs, batches):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.weights)
        with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
            w = csv.writer(f)
            w.writerow(['Epoch', 'NLL', 'Loss'])
            f.flush()
            beta = 1e-2
            loss = None
            nll = None
            for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                for batch in it:
                    _nll, _loss = self.train_fun()
                    if not loss:
                        loss = _loss
                    if not nll:
                        nll = _nll
                    loss = (beta * _loss) + ((1. - beta) * loss)
                    nll = (beta * _nll) + ((1. - beta) * nll)
                    it.desc = "Epoch {} NLL {:.04f} Loss {:.04f}".format(epoch, np.asscalar(nll), np.asscalar(loss))
                w.writerow([epoch, np.asscalar(nll), np.asscalar(loss)])
                f.flush()
                z = self.encodings_fun()  # (n, z_units)
                np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), z)
                save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.weights)
        return self.val_fun()


def run_baseline(z_ks, iters, output_path, cooccurrence, epochs, batches, regularizer=None):
    nlls = []
    losses = []
    for z_units in z_ks:
        _nlls = []
        _losses = []
        for i in range(iters):
            iter_path = "{}/z-{}/iter-{}".format(output_path, z_units, iters)
            model = BaselineModel(
                cooccurrence=cooccurrence,
                z_units=z_units,
                opt=Adam(1e-3),
                regularizer=regularizer
            )
            nll, loss = model.train(outputpath=iter_path,
                                    epochs=epochs,
                                    batches=batches)
            _nlls.append(nll)
            _losses.append(loss)
        nlls.append(_nlls)
        losses.append(_losses)
    np.savez("{}.npz".format(output_path), z_ks=np.array(z_ks), nlls=np.array(nlls), losses=np.array(losses))
