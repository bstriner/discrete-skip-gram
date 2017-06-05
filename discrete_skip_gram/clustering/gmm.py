import os
import csv
import itertools

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def cluster_recursive(words, code, depth):
    if depth == 0:
        return [(code, word) for word in words]
    else:
        if len(words) == 1:
            return cluster_recursive(words, code+[0], depth-1)
        else:
            gmm = GaussianMixture(n_components=2, n_init=3)
            z = np.stack(w['z'] for w in words)
            gmm.fit(z)
            p = gmm.predict_proba(z)
            score = p[:,0]-p[:,1]
            order = np.argsort(score)
            orderedwords = [words[order[j]] for j in range(order.shape[0])]
            half = int(len(words)/2)
            a = orderedwords[:half]
            b = orderedwords[half:]
            ar = cluster_recursive(a, code+[0], depth-1)
            br = cluster_recursive(b, code+[1], depth-1)
            return ar + br

def cluster_gmm(z, depth):
    words = [{'idx': i, 'z': z[i, :]} for i in range(z.shape[0])]
    coded = cluster_recursive(words, [], depth)
    coded.sort(key=lambda _x: _x[1]['idx'])
    encodings = [np.array(_x[0]) for _x in coded]
    enc = np.stack(encodings)
    return enc