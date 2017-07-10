import csv
import os

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from discrete_skip_gram.skipgram.tensor_util import softmax_nd
from .tensor_util import save_weights, load_latest_weights


class CategoricalNNModel(object):
    def __init__(self, x_k, z_k, opt, regularizer=None):
        scale = 1e-1

        # parameters
        initial_weight = np.random.uniform(-scale, scale, (x_k, z_k)).astype(np.float32)
        pz_weight = theano.shared(initial_weight, name="weight")  # (x_k, z_k)
        initial_py = np.random.uniform(-scale, scale, (z_k, x_k)).astype(np.float32)  # (z_k, x_k)
        py_weight = theano.shared(initial_py, name='py')  # (z_k, x_k)
        params = [pz_weight, py_weight]

        # samples
        data = T.imatrix()  # (n,2)
        a = data[:, 0]
        b = data[:, 1]

        parts = []
        for x, y in [[a, b], [b, a]]:
            # p_z
            p_z = softmax_nd(pz_weight[x, :])  # (n, z_k)

            # p_y
            p_y = softmax_nd(py_weight)  # (z_k, x_k)
            p_yt = T.transpose(p_y, (1, 0))[y, :]  # (n, z_k)
            eps = 1e-8
            nll_y = -T.log(p_yt + eps)  # (n, z_k)
            losspart = T.sum(p_z * nll_y, axis=1)  # (n,)
            parts.append(losspart)

        meannll = (parts[0] + parts[1]) / 2.  # (n,)
        nll = T.mean(meannll)  # scalar
        # loss
        loss = nll
        if regularizer:
            for p in params:
                loss += regularizer(p)
        updates = opt.get_updates(params, {}, loss)
        train = theano.function([data], [nll, loss], updates=updates)

        encs = softmax_nd(pz_weight)
        encodings = theano.function([], encs)
        self.train_fun = train
        self.encodings_fun = encodings
        self.all_weights = params + opt.weights

    def train_batch(self, dataset, batch_size=32):
        nll = 0.
        loss = 0.
        np.random.shuffle(dataset)
        n = dataset.shape[0]
        batch_count = int(np.ceil(float(n) / float(batch_size)))
        for batch in tqdm(range(batch_count)):
            i1 = batch * batch_size
            i2 = (batch + 1) * batch_size
            if i2 > n:
                i2 = n
            b = dataset[i1:i2, :]
            _nll, _loss = self.train_fun(b)
            nll += _nll
            loss += _loss
        fn = float(n)
        return nll/fn, loss/fn

    def train(self, outputpath, epochs, batches, batch_size, dataset):
        initial_epoch = load_latest_weights(outputpath, r'model-(\d+).h5', self.all_weights)
        ds = np.copy(dataset)
        with open(os.path.join(outputpath, 'history.csv'), 'ab') as f:
            w = csv.writer(f)
            w.writerow(['Epoch', 'Loss', 'NLL'])
            f.flush()
            for epoch in tqdm(range(initial_epoch, epochs), desc="Training"):
                it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                for batch in it:
                    nll, loss = self.train_batch(dataset=ds, batch_size=batch_size)
                    it.desc = "Epoch {} Loss {:.4f} NLL {:.4f}".format(epoch, np.asscalar(loss), np.asscalar(nll))
                w.writerow([epoch, loss, nll])
                f.flush()
                enc = self.encodings_fun()  # (n, x_k)
                np.save(os.path.join(outputpath, 'probabilities-{:08d}.npy'.format(epoch)), enc)
                z = np.argmax(enc, axis=1)  # (n,)
                np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), z)
                save_weights(os.path.join(outputpath, 'model-{:08d}.h5'.format(epoch)), self.all_weights)
