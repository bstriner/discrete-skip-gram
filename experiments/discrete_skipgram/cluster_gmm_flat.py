import csv

import numpy as np
from tqdm import tqdm

from discrete_skip_gram.clustering.gmm import cluster_gmm_flat
from discrete_skip_gram.models.util import latest_model
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.validation import validate_encoding_flat


def stats(nlls):
    return [np.mean(nlls), np.min(nlls), np.max(nlls), np.std(nlls)]


def cluster_val(z, z_k, co, iters):
    nlls = []
    for i in tqdm(range(iters), desc="zk={}".format(z_k)):
        enc = cluster_gmm_flat(z, z_k)
        loss = validate_encoding_flat(enc=enc, co=co, z_k=z_k)
        nlls.append(loss)
    return nlls


def main():
    output_path = "output/cluster_gmm_flat.csv"
    path = "output/skipgram_baseline"
    file, epoch = latest_model(path, "encodings-(\\d+).npy", fail=True)
    print "Loading epoch {}: {}".format(epoch, file)
    z = np.load(file)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    co = cooccurrence / np.sum(cooccurrence, axis=None)
    iters = 5
    zks = [(2 ** i) for i in range(11)]
    with open(output_path, 'wb') as f:
        w = csv.writer(f)
        w.writerow(
            ['Zk (epoch {})'.format(epoch), 'Mean', 'Min', 'Max', 'Std'] + ['Iter {}'.format(i) for i in range(iters)])
        for zk in tqdm(zks):
            nlls = cluster_val(z=z, z_k=zk, co=co, iters=iters)
            w.writerow([zk] + stats(nlls) + nlls)


if __name__ == "__main__":
    main()
