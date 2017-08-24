import itertools
import os

import numpy as np
from tqdm import tqdm

from .util import latest_file


class GreedyModel(object):
    def __init__(self, cooccurrence, z_k, repeats=1):
        self.repeats = repeats
        n = cooccurrence.shape[0]
        z = np.random.random_integers(0, z_k-1, (n,))
        self.z = z
        self.z_k = z_k
        self.n = n

        h = cooccurrence.astype(np.float32)
        p = h / np.sum(h, axis=None)
        self.p = p
        self.val_fun = self.make_val()

    def make_val(self):
        import theano
        import theano.tensor as T
        eps = 1e-9
        input_enc = T.ivector()
        c = T.zeros((self.z_k, self.n), dtype='float32')  # (z_k, n)
        c = T.set_subtensor(c[input_enc, T.arange(c.shape[1])], 1)
        p_yz = T.dot(c, T.constant(self.p, name='p'))  # (z_k, n) x (n, n) = (z_k, n)
        marg = T.sum(p_yz, axis=1, keepdims=True)
        cond = p_yz / (eps + marg)
        ent = -T.sum(p_yz * T.log(eps + cond), axis=None)
        return theano.function([input_enc], ent)

    def gen_choices(self):
        for i in range(self.n):
            for j in range(self.z_k):
                yield i, j

    def val_choice(self, choice, z):
        nz = np.copy(z)
        if self.repeats > 1:
            for i, j in choice:
                nz[i] = j
        else:
            i, j = choice
            nz[i] = j
        return nz, self.calc(nz)

    def train_batch(self):
        choices = self.gen_choices()
        if self.repeats > 1:
            choices = itertools.combinations(choices, repeats=self.repeats)

        bze = self.calc(self.z)
        bz = self.z
        flag = False
        t = tqdm(list(choices), desc="Batch")
        for choice in t:
            nz, nze = self.val_choice(choice, bz)
            if nze < bze:
                bze = nze
                bz = nz
                flag = True
            t.desc = "NLL [{:.03f}]".format(np.asscalar(bze))
        self.z = bz
        return flag

    def train(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        path, initial_epoch = latest_file(output_path, r'z-(\d+).npy')
        if path:
            self.z = np.load(path)
        epoch = initial_epoch
        flag = True
        t = tqdm(desc="Training")
        if epoch > 0:
            t.update(epoch)
        while (flag):
            flag = self.train_batch()
            np.save(os.path.join(output_path, 'z-{:08d}.npy'.format(epoch)))
            t.update(1)
            epoch += 1

    def calc(self, enc):
        return self.calc_T(enc)
        #return self.calc_np(enc)

    def calc_T(self, enc):
        return self.val_fun(enc)

    def calc_np(self, enc):
        c = np.zeros((self.z_k, self.n))
        assert len(enc.shape) == 1
        c[enc, np.arange(c.shape[1])] = 1
        p_yz = np.dot(c, self.p)

        eps = 1e-9
        marg = np.sum(p_yz, axis=1, keepdims=True)
        cond = p_yz / (eps + marg)
        ent = -np.sum(p_yz * np.log(eps + cond), axis=None)
        return ent
