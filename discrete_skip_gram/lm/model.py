import csv

import numpy as np
from tqdm import tqdm

from ..tensor_util import load_latest_weights, save_weights
from ..util import generate_batch_indices
from ..util import make_dir


class LanguageModel(object):
    def __init__(self, weights, train_headers, val_headers):
        self.weights = weights
        self.train_headers = train_headers
        self.val_headers = val_headers

    def generate(self, samples, depth):
        # override in child
        pass

    def save_output(self, output_path, epoch, xvalid, xtest):
        # override in child
        pass

    def train_batchx(self, x, **kwargs):
        # override in child
        return 1

    def batch_data(self,xtrain, depth=35, batch_size=64):
        max_idx = xtrain.shape[0] - depth
        idx = np.random.random_integers(low=0, high=max_idx, size=(batch_size,))
        ids = (idx.reshape((-1, 1))) + (np.arange(depth).reshape((1, -1)))
        xsel = xtrain[ids]
        return xsel

    def train_batch(self, xtrain, depth=35, batch_size=64, **kwargs):
        xsel = self.batch_data(xtrain=xtrain, depth=depth, batch_size=batch_size)
        return self.train_batchx(xsel, **kwargs)

    def train_batches(self, xtrain, label, batches=1024, **kwargs):
        it = tqdm(range(batches), desc=label)
        k = len(self.train_headers)
        data = [[] for _ in range(k)]
        for _ in it:
            datum = self.train_batch(xtrain, **kwargs)
            for i in range(k):
                data[i].append(datum[i])
            strs = [label]
            for i in range(k):
                strs.append('Mean {} {:.4f}'.format(self.train_headers[i], np.asscalar(np.mean(data[i]))))
                strs.append('Current {} {:.4f}'.format(self.train_headers[i], np.asscalar(datum[i])))
            it.desc = " ".join(strs)
        return [np.asscalar(np.mean(d)) for d in data]

    def train(self,
              output_path,
              epochs,
              xtrain,
              xvalid,
              xtest,
              **kwargs):
        make_dir(output_path)
        initial_epoch = load_latest_weights(output_path, fmt=r'model-(\d+).h5', weights=self.weights)
        if initial_epoch < epochs:
            csv_path = '{}/history.csv'.format(output_path)
            with open(csv_path, 'ab') as f:
                w = csv.writer(f)
                headers = ['Epoch'] + self.train_headers
                for h in self.val_headers:
                    headers.append('Validation {}'.format(h))
                for h in self.val_headers:
                    headers.append('Test {}'.format(h))
                w.writerow(headers)
                f.flush()
                it = tqdm(range(initial_epoch, epochs), desc='Training')
                for epoch in it:
                    train_stats = self.train_batches(xtrain=xtrain,
                                                     label="Epoch {}".format(epoch),
                                                     **kwargs)
                    self.save_output(output_path=output_path,
                                     epoch=epoch,
                                     xvalid=xvalid,
                                     xtest=xtest)
                    w.writerow([epoch] +
                               train_stats +
                               self.validate(xvalid, **kwargs) +
                               self.validate(xtest, **kwargs))
                    f.flush()
                    save_weights(path='{}/model-{:08d}.h5'.format(output_path, epoch),
                                 weights=self.weights)

    def validate(self, x, batch_size=64, depth=35, **kwargs):
        # calc perplexity on test set
        stack = []
        n = x.shape[0]
        idx = list(generate_batch_indices(n=n - depth + 1, batch_size=batch_size))
        for idx0, idx1 in tqdm(idx, desc='Validating'):
            i1 = np.arange(idx0, idx1).reshape((-1, 1))
            i2 = np.arange(depth).reshape((1, -1))
            i = i1 + i2
            xb = x[i]
            nll = self.nll_fun(xb)
            stack.append(nll)
        nll = np.concatenate(stack, axis=0)
        p0 = nll[0, :]  # (d,)
        p1 = nll[1:, depth - 1]  # (n-d,)
        nllsel = np.concatenate((p0, p1), axis=0)
        assert nllsel.shape[0] == x.shape[0]
        avgnll = np.mean(nllsel)
        tqdm.write("Validation: {}, {}".format(avgnll, np.power(2, avgnll)))
        return [np.asscalar(avgnll), np.asscalar(np.power(2, avgnll))]
