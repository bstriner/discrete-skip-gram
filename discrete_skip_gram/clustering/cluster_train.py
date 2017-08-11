import numpy as np
from tqdm import tqdm
import os
from ..util import make_path


def cluster_iters(z, iters, val_fun, z_k, cooccurrence):
    nlls = []
    for _ in tqdm(range(iters), desc='Clustering'):
        nll = val_fun(z=z, z_k=z_k, cooccurrence=cooccurrence)
        nlls.append(nll)
    return np.stack(nlls)


def cluster_dir(input_path, bzks, iters, z_k, cooccurrence, val_fun, desc='Training'):
    all_nlls = []
    for bzk in tqdm(bzks, desc=desc):
        z_path = "{}/z-{}-embeddings.npy".format(input_path, bzk)
        z = np.load(z_path)  # (iters, n, z_units)
        biters = z.shape[0]
        nlls = []
        for biter in tqdm(range(biters), desc="Baseline iterations"):
            zt = z[biter, :, :]  # (n, z_units)
            nll = cluster_iters(z=zt,
                                iters=iters,
                                val_fun=val_fun,
                                z_k=z_k,
                                cooccurrence=cooccurrence)
            nlls.append(nll)
        all_nlls.append(np.stack(nlls))
    return np.stack(all_nlls)


def train_clusters(
        output_path,
        input_path, bzks, iters, z_k, cooccurrence, val_fun, desc='Training'):
    if os.path.exists(output_path):
        return np.load(output_path)
    else:
        make_path(output_path)
        nlls = cluster_dir(input_path=input_path,
                           bzks=bzks,
                           iters=iters,
                           z_k=z_k,
                           desc=desc,
                           cooccurrence=cooccurrence,
                           val_fun=val_fun)
        np.save(output_path, nlls)
        return nlls


def train_cluster_battery(output_path,
                          input_paths, labels, bzks, iters, z_k, cooccurrence, val_fun, desc):
    kwdata = {'bzks': np.array(bzks)}
    for input_path, label in zip(input_paths, labels):
        kwdata[label] = train_clusters(input_path=input_path,
                                       output_path='{}/{}.npy'.format(output_path, label),
                                       bzks=bzks,
                                       iters=iters,
                                       z_k=z_k,
                                       desc='Clustering {} {}'.format(desc, label),
                                       cooccurrence=cooccurrence,
                                       val_fun=val_fun)
    # (bzks, biters, iters)
    np.savez('{}.npz'.format(output_path), **kwdata)
