import numpy as np
from sklearn.mixture import GaussianMixture

from ..flat_validation import validate_encoding_flat


def cluster_hgmm_recursive(words, code, depth):
    if depth == 0:
        return [(code, word) for word in words]
    else:
        if len(words) == 1:
            return cluster_hgmm_recursive(words, code + [0], depth - 1)
        else:
            gmm = GaussianMixture(n_components=2, n_init=5)
            z = np.stack(w['z'] for w in words)
            gmm.fit(z)
            p = gmm.predict_proba(z)
            eps = 1e-9
            score = np.log(eps + p[:, 0]) - np.log(eps + p[:, 1])
            order = np.argsort(score)
            orderedwords = [words[order[j]] for j in range(order.shape[0])]
            half = int(len(words) / 2)
            a = orderedwords[:half]
            b = orderedwords[half:]
            ar = cluster_hgmm_recursive(a, code + [0], depth - 1)
            br = cluster_hgmm_recursive(b, code + [1], depth - 1)
            return ar + br


def cluster_hgmm(z, depth):
    words = [{'idx': i, 'z': z[i, :]} for i in range(z.shape[0])]
    coded = cluster_hgmm_recursive(words, [], depth)
    coded.sort(key=lambda _x: _x[1]['idx'])
    encodings = [np.array(_x[0]) for _x in coded]
    enc = np.stack(encodings)
    return enc


def validate_cluster_hgmm(z, z_k, cooccurrence):
    depth = int(np.log2(z_k))
    assert 2 ** depth == z_k
    enc = cluster_hgmm(z, depth)
    m = np.power(2, np.arange(depth)).reshape((1, -1))
    encf = np.sum(enc * m, axis=1)
    nll = validate_encoding_flat(cooccurrence=cooccurrence, enc=encf)
    return nll


def cluster_gmm_flat(z, z_k, n_init=1):
    gmm = GaussianMixture(n_components=z_k, n_init=n_init)
    gmm.fit(z)
    p = gmm.predict_proba(z)  # (x_k, z_k)
    enc = np.argmax(p, axis=1)  # (x_k,)
    return enc


def validate_cluster_gmm(z, z_k, cooccurrence):
    enc = cluster_gmm_flat(z, z_k)
    nll = validate_encoding_flat(cooccurrence=cooccurrence, enc=enc)
    return nll


def cluster_balanced_gmm(z, z_k, n_init=1):
    assert z_k == 2
    gmm = GaussianMixture(n_components=2, n_init=n_init)
    gmm.fit(z)
    p = gmm.predict_proba(z)  # (x_k, 2)
    eps = 1e-9
    h = np.log(eps + p[:, 0]) - np.log(eps + p[:, 1])  # (x_k,)
    order = np.argsort(h)  # (x_k,)
    mid = int(order.shape[0] / 2)
    top = order[:mid]
    enc = np.zeros((z.shape[0],), dtype='int32')
    enc[top] = 1
    return enc


def validate_balanced_gmm(z, z_k, cooccurrence):
    enc = cluster_balanced_gmm(z, z_k)
    nll = validate_encoding_flat(cooccurrence=cooccurrence, enc=enc)
    return nll
