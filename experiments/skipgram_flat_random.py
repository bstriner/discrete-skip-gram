import csv

import numpy as np
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.validation import validate_encoding_flat
from tqdm import tqdm

from discrete_skip_gram.util import make_path


def random_baseline(csv_path, iters, z_k, val, x_k):
    nlls = []
    make_path(csv_path)
    with open(csv_path, 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Iteration', 'Loss'])
        for i in tqdm(range(iters), desc="Zk={}".format(z_k)):
            enc = np.random.random_integers(0, z_k - 1, (x_k,))
            loss = val(enc=enc, z_k=z_k)
            nlls.append(loss)
            w.writerow([i, loss])
    return nlls


def main():
    iters = 100
    z_ks = [2 ** i for i in range(11)]
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    x_k = cooccurrence.shape[0]
    output_path = "output/skipgram_flat_random"
    val = validate_encoding_flat(cooccurrence)
    kwargs = {}
    with open("{}.csv".format(output_path), 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Zk', 'Mean', 'Min', 'Max', 'Std'] + ["Iter {}".format(i) for i in range(iters)])
        for z_k in tqdm(z_ks, desc="Testing"):
            csv_path = "{}/{}.csv".format(output_path, z_k)
            nlls = random_baseline(csv_path, iters, z_k, val=val, x_k=x_k)
            kwargs["z_{}".format(z_k)] = np.array(nlls)
            w.writerow([z_k, np.mean(nlls), np.min(nlls), np.max(nlls), np.std(nlls)] + nlls)
    np.savez("{}.npz", z_ks=np.array(z_ks), **kwargs)


if __name__ == "__main__":
    main()
