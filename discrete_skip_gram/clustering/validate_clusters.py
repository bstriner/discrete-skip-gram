import csv

import numpy as np
from tqdm import tqdm
from discrete_skip_gram.skipgram.util import make_path


def stats(nlls):
    return [np.mean(nlls), np.min(nlls), np.max(nlls), np.std(nlls)]


def cluster_val(z, z_k, iters, clustering, validation):
    nlls = []
    for i in tqdm(range(iters), desc="zk={}".format(z_k)):
        enc = clustering(z=z, z_k=z_k).astype(np.int32)
        loss = validation(enc, z_k)
        nlls.append(loss)
    return nlls


def validate_clusters(output_path, z, clustering, zks, iters,
                      validation):
    with open(output_path, 'wb') as f:
        w = csv.writer(f)
        w.writerow(
            ['Zk', 'Mean', 'Min', 'Max', 'Std'] + ['Iter {}'.format(i) for i in range(iters)])
        for zk in tqdm(zks):
            nlls = cluster_val(z=z, z_k=zk, iters=iters, clustering=clustering, validation=validation)
            w.writerow([zk] + stats(nlls) + nlls)


def validate_clustering(output_path, z, clustering, z_k, iters,
                        validation):
    make_path(output_path)
    with open(output_path, 'wb') as f:
        w = csv.writer(f)
        w.writerow(
            ['Iteration', 'NLL'])
        nlls = cluster_val(
            z=z,
            z_k=z_k,
            iters=iters,
            clustering=clustering,
            validation=validation)
        for i, nll in enumerate(nlls):
            w.writerow([i, nll])
        w.writerow(['Mean', np.mean(nlls)])
        w.writerow(['Min', np.min(nlls)])
        w.writerow(['Max', np.max(nlls)])
        w.writerow(['Std', np.std(nlls)])
        return nlls
