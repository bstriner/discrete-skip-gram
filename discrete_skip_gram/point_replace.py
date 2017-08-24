import os

import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from .util import latest_file


class PointReplaceModel(object):
    def __init__(self, cooccurrence, z_k):
        eps = 1e-9
        self.z_k = z_k
        n = cooccurrence.shape[0]
        self.n = n
        h = cooccurrence.astype(np.float32)
        p = h / np.sum(h, axis=None)
        pc = T.constant(p, name="p")

        z_init = np.random.random_integers(0, z_k - 1, (n,))
        zt = theano.shared(z_init)
        self.z_shared = zt

        idx = T.iscalar(name='idx')
        c = T.zeros((z_k, n), dtype='float32')  # (z_k, n)
        c = T.set_subtensor(c[zt, T.arange(c.shape[1])], 1)
        c = T.set_subtensor(c[zt[idx], idx], 0)

        p_yz0 = T.dot(c, pc)  # (z_k, n) x (n, n) = (z_k, n)
        marg0 = T.sum(p_yz0, axis=1, keepdims=True)
        cond0 = p_yz0 / (eps + marg0)
        ent0 = -T.sum(p_yz0 * T.log(eps + cond0), axis=1)  # (z_k,)
        sum0 = T.sum(ent0, axis=0)

        p_yz1 = p_yz0 + (pc[idx, :].dimshuffle(('x', 0)))
        marg1 = T.sum(p_yz1, axis=1, keepdims=True)
        cond1 = p_yz1 / (eps + marg1)
        ent1 = -T.sum(p_yz1 * T.log(eps + cond1), axis=1)  # (z_k,)

        ed = ent1 - ent0  # (z_k)
        sel = T.argmin(ed)  # (scalar,)
        ztn = T.set_subtensor(zt[idx], sel)
        entn = sum0 + (ed[sel])
        changed = T.neq(sel, zt[idx])
        updates = [(zt, ztn)]

        self.train_fun = theano.function([idx], [entn, changed], updates=updates)

        c = T.zeros((z_k, n), dtype='float32')  # (z_k, n)
        c = T.set_subtensor(c[zt, T.arange(c.shape[1])], 1)
        p_yz = T.dot(c, pc)  # (z_k, n) x (n, n) = (z_k, n)
        marg = T.sum(p_yz, axis=1, keepdims=True)
        cond = p_yz / (eps + marg)
        ent = -T.sum(p_yz * T.log(eps + cond), axis=None)

        self.val_fun = theano.function([], ent)

    def train_batch(self):
        t = tqdm(range(self.n), desc='Batch')
        flag = False
        for i in t:
            entn, changed = self.train_fun(i)
            if changed > 0:
                flag = True
            t.desc = 'Batch [{:.04f}]{}'.format(np.asscalar(entn), '*' if flag else '')
        return flag

    def train(self, output_path, frequency=1):
        result_path = "{}.npy".format(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        path, last_epoch = latest_file(output_path, r'z-(\d+).npy')
        if path:
            K.set_value(self.z_shared, np.load(path))
            epoch = last_epoch + 1
        else:
            epoch = 0
        if not os.path.exists(result_path):
            flag = True
            t = tqdm(desc="Training")
            if epoch > 0:
                t.update(epoch)
            while flag:
                flag = self.train_batch()
                if (not flag) or (epoch % frequency == 0):
                    np.save(os.path.join(output_path, 'z-{:08d}.npy'.format(epoch)), K.get_value(self.z_shared))
                t.update(1)
                epoch += 1
            val = self.val_fun()
            np.savetxt("{}.txt".format(output_path), val.reshape((1, 1)))
            np.save(result_path, val)
        return self.val_fun()


def train_battery(output_path, iters, cooccurrence, z_k):
    vs = []
    for i in range(iters):
        path = os.path.join(output_path, "iter-{}".format(i))
        m = PointReplaceModel(cooccurrence=cooccurrence, z_k=z_k)
        v = m.train(output_path=path)
        vs.append(v)
    vs = np.array(vs)
    np.save("{}.npy".format(output_path), vs)
