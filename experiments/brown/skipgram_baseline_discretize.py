import os
import csv
import itertools

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from dataset import load_dataset


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


from discrete_skip_gram.models.util import latest_model


def main():
    ds = load_dataset()
    encoding_path = "output/brown/skipgram_baseline"
    encoding_file, epoch = latest_model(encoding_path, "encodings-(\\d+).npy")
    print "Loading {}".format(encoding_file)
    with open(encoding_file, 'rb') as f:
        z = np.load(f)
    c = AgglomerativeClustering(n_clusters=1, compute_full_tree=True,
                                linkage='ward')
    c.fit(z)
    d = c.children_
    nodes = {}
    n = z.shape[0]
    for i in range(n - 1):
        l = d[i, 0]
        r = d[i, 1]
        if l > n - 1:
            l = nodes[l - n]
        if r > n - 1:
            r = nodes[r - n]
        nodes[i] = (l, r)
    root = nodes[n - 2]
    depth = 12
    encodings = []
    for p in itertools.product([0, 1], repeat=depth):
        enc = np.array(p)
        words = get_words(p, root)
        for word in words:
            encodings.append((word, enc))

    encodings.sort(key=lambda _x: _x[0])
    arr = np.stack([e[1] for e in encodings], axis=0).astype(np.int32)
    print "Z shape: {}".format(z.shape)
    print "Array shape: {}".format(arr.shape)

    output_path = "output/brown/skipgram_baseline_discrete"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.save('{}/encoding.npy'.format(output_path), arr)
    with open('{}/encoding.csv'.format(output_path), 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Word', 'Encoding'] + ['Cat {}'.format(i) for i in range(depth)])
        for i in range(arr.shape[0]):
            word = ds.get_word(i)
            enc = arr[i, :]
            enca = [enc[j] for j in range(enc.shape[0])]
            encf = "".join(chr(ord('a') + enc[j]) for j in range(enc.shape[0]))
            w.writerow([i, word, encf] + enca)


if __name__ == "__main__":
    main()
