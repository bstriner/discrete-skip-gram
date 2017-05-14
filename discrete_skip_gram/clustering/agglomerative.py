import os
import csv
import itertools

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def node_words(n):
    if isinstance(n, (tuple, list)):
        return list(itertools.chain.from_iterable(node_words(m) for m in n))
    else:
        return [n]


def get_words(ps, root):
    n = root
    for p in ps:
        if isinstance(n, (tuple, list)):
            n = n[p]
        elif p == 0:
            n = n
        else:
            return []
    return node_words(n)


def flatten_words(n):
    if isinstance(n, (list, tuple)):
        return list(itertools.chain.from_iterable(flatten_words(m) for m in n))
    else:
        return [n]


def cluster_agglomerative(z, depth):
    assert z.ndim == 2
    n = z.shape[0]
    c = AgglomerativeClustering(n_clusters=1, compute_full_tree=True,
                                linkage='ward')
    c.fit(z)
    d = c.children_
    nodes = {}
    for i in range(n - 1):
        l = d[i, 0]
        r = d[i, 1]
        if l > n - 1:
            l = nodes[l - n]
        if r > n - 1:
            r = nodes[r - n]
        nodes[i] = (l, r)
    root = nodes[n - 2]
    encodings = []
    for p in itertools.product([0, 1], repeat=depth):
        enc = np.array(p)
        words = get_words(p, root)
        for word in words:
            encodings.append((word, enc))

    encodings.sort(key=lambda _x: _x[0])
    clusters = np.stack([e[1] for e in encodings], axis=0).astype(np.int32)
    print "Z shape: {}".format(z.shape)
    print "Array shape: {}".format(clusters.shape)

    return clusters
