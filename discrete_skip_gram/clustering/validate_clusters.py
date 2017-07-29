import csv

import numpy as np
from tqdm import tqdm


def stats(nlls):
    return [np.mean(nlls), np.min(nlls), np.max(nlls), np.std(nlls)]


def cluster_val(z, z_k, co, iters, clustering, validation):
    nlls = []
    for i in tqdm(range(iters), desc="zk={}".format(z_k)):
        enc = clustering(z=z, z_k=z_k).astype(np.int32)
        loss = validation(enc, z_k)
        nlls.append(loss)
    return nlls


def validate_clusters(output_path, z, cooccurrence, clustering, zks, iters,
                      validation):
    cooccurrence = cooccurrence.astype(np.float32)
    co = cooccurrence / np.sum(cooccurrence, axis=None)
    with open(output_path, 'wb') as f:
        w = csv.writer(f)
        w.writerow(
            ['Zk', 'Mean', 'Min', 'Max', 'Std'] + ['Iter {}'.format(i) for i in range(iters)])
        for zk in tqdm(zks):
            nlls = cluster_val(z=z, z_k=zk, co=co, iters=iters, clustering=clustering, validation=validation)
            w.writerow([zk] + stats(nlls) + nlls)
