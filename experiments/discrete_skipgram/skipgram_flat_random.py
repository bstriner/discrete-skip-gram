# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

import csv

import numpy as np
from tqdm import tqdm

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence


def validate_encoding_flat(enc, co, z_k, x_k, eps=1e-9):
    m = np.zeros((z_k, x_k))  # zk, xk
    m[enc, np.arange(x_k)] = 1
    p = np.dot(m, co)  # (z_k, x_k) * (x_k, x_k) = z_k, x_k
    marg = np.sum(p, axis=1, keepdims=True)
    cond = p / (marg + eps)
    nll = np.sum(cond * -np.log(eps + cond), axis=1, keepdims=True)  # (z_k, 1)
    loss = np.asscalar(np.sum(nll * marg, axis=None))
    return loss

def flat_baseline(outputpath, iters, z_k):
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    x_k = cooccurrence.shape[0]
    co = cooccurrence / np.sum(cooccurrence, axis=None)
    nlls = []
    with open(outputpath, 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Loss'])
        for i in tqdm(range(iters), desc="Zk={}".format(z_k)):
            enc = np.random.random_integers(0, z_k - 1, (x_k,))
            loss = validate_encoding_flat(enc=enc, co=co, z_k=z_k, x_k=x_k)
            nlls.append(loss)
            w.writerow([i, loss])
    return nlls


def main():
    iters = 100
    vals = []
    #zks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    zks = [2**i for i in range(11)]
    with open("output/skipgram_flat_random.csv", 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Zk', 'Mean', 'Min', 'Max', 'Std'] + ["Iter {}".format(i) for i in range(iters)])
        for z_k in tqdm(zks, desc="Testing"):
            outputpath = "output/skipgram_flat_random-{}.csv".format(z_k)
            val = flat_baseline(outputpath, iters, z_k)
            vals.append(val)
            w.writerow([z_k, np.mean(val), np.min(val), np.max(val), np.std(val)] + val)


if __name__ == "__main__":
    main()
