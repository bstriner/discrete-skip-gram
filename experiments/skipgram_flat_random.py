import csv

import numpy as np
from discrete_skip_gram.flat_validation import validate_encoding_flat
from tqdm import tqdm

from discrete_skip_gram.util import make_path


def random_baseline(csv_path, iters, z_k, x_k, cooccurrence):
    nlls = []
    make_path(csv_path)
    with open(csv_path, 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Iteration', 'Loss'])
        for i in tqdm(range(iters), desc="Zk={}".format(z_k)):
            enc = np.random.random_integers(0, z_k - 1, (x_k,))
            loss = validate_encoding_flat(cooccurrence=cooccurrence, enc=enc)
            nlls.append(loss)
            w.writerow([i, loss])
    return np.array(nlls)


def main():
    iters = 100
    z_ks = [2 ** (i + 1) for i in range(10)]
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    x_k = cooccurrence.shape[0]
    output_path = "output/skipgram_flat_random"
    all_nlls = []
    with open("{}.csv".format(output_path), 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Zk', 'Mean', 'Min', 'Max', 'Std'] + ["Iter {}".format(i) for i in range(iters)])
        for z_k in tqdm(z_ks, desc="Testing"):
            csv_path = "{}/{}.csv".format(output_path, z_k)
            nlls = random_baseline(csv_path=csv_path,
                                   iters=iters,
                                   z_k=z_k,
                                   x_k=x_k,
                                   cooccurrence=cooccurrence)
            all_nlls.append(nlls)
            w.writerow([z_k, np.mean(nlls), np.min(nlls), np.max(nlls), np.std(nlls)] + [n for n in nlls])
    all_nlls = np.stack(all_nlls)
    np.savez("{}.npz".format(output_path),
             nlls=all_nlls,
             z_ks=np.array(z_ks))


if __name__ == "__main__":
    main()
