import numpy as np
from tqdm import tqdm


def cluster_iters(z, iters, val_fun, z_k, cooccurrence):
    nlls = []
    for _ in tqdm(range(iters), desc='Clustering'):
        nll = val_fun(z=z, z_k=z_k, cooccurrence=cooccurrence)
        nlls.append(nll)
    return np.stack(nlls)


def cluster_dir(input_path, bzks, iters, z_k, cooccurrence, val_fun):
    all_nlls = []
    for bzk in tqdm(bzks, desc="BZKs"):
        z_path = "{}/z-{}-embeddings.npy".format(input_path, bzk)
        z = np.load(z_path)  # (iters, n, z_units)
        biters = z.shape[0]
        nlls = []
        for biter in range(biters):
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
        input_path, bzks, iters, z_k, cooccurrence, val_fun):
    nlls = cluster_dir(input_path=input_path,
                       bzks=bzks,
                       iters=iters,
                       z_k=z_k,
                       cooccurrence=cooccurrence,
                       val_fun=val_fun)
    np.save(output_path, nlls)
