import os

import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from .util import latest_file


class EMModel(object):
    def __init__(self, cooccurrence, z_k):
        eps = 1e-9
        self.z_k = z_k
        n = cooccurrence.shape[0]
        self.n = n
        h = cooccurrence.astype(np.float32)
        p = h / np.sum(h, axis=None)
        pc = T.constant(p, name="p")

        z_init = np.random.random_integers(0, z_k - 1, (n,)).astype(np.int32)
        z = theano.shared(z_init, name="z")
        self.z = z

        c = T.zeros((z_k, n), dtype='float32')  # (z_k, n)
        c = T.set_subtensor(c[z, T.arange(c.shape[1])], 1)

        pyz = T.dot(c, pc)  # (z_k, x_k)
        marg = T.sum(pyz, axis=1, keepdims=True)
        cond = pyz / (marg + eps)
        nll = -T.sum(pyz * T.log(eps + cond), axis=None)  # scalar

        nllyzr = T.transpose(-T.log(eps + cond), (1, 0))  # (x_k, z_k)
        losses = T.dot(pc, nllyzr)  # (x_k, z_k)
        nz = T.cast(T.argmin(losses, axis=1), 'int32')  # (x_k,)
        updates = [(z, nz)]

        flag = T.gt(T.sum(T.neq(z, nz)), 0)

        self.train_fun = theano.function([], [nll, flag], updates=updates)
        self.val_fun = theano.function([], nll)

    def train(self, output_path, frequency=1):
        result_path = "{}.npy".format(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        path, last_epoch = latest_file(output_path, r'z-(\d+).npy')
        if path:
            K.set_value(self.z, np.load(path))
            epoch = last_epoch + 1
        else:
            epoch = 0
        if not os.path.exists(result_path):
            flag = True
            t = tqdm(desc="Training")
            if epoch > 0:
                t.update(epoch)
            while flag:
                nll, flag = self.train_fun()
                print("NLL: {}, {}".format(nll, flag))
                t.desc = 'Training [{:.04f}]'.format(np.asscalar(nll))
                t.update(1)
                if (not flag) or (epoch % frequency == 0):
                    np.save(os.path.join(output_path, 'z-{:08d}.npy'.format(epoch)), K.get_value(self.z))
                epoch += 1
            val = self.val_fun()
            np.savetxt("{}.txt".format(output_path), val.reshape((1, 1)))
            np.save(result_path, val)
        return self.val_fun()


def train_battery(output_path, iters, cooccurrence, z_k):
    vs = []
    for i in tqdm(range(iters), desc='Metaiteration'):
        path = os.path.join(output_path, "iter-{}".format(i))
        m = EMModel(cooccurrence=cooccurrence, z_k=z_k)
        v = m.train(output_path=path)
        vs.append(v)
    vs = np.array(vs)
    np.save("{}.npy".format(output_path), vs)
    for i, v in enumerate(vs):
        print("Iteration {}: {}".format(i, v))
    print("Mean: {}, Std: {}".format(np.mean(vs), np.std(vs)))