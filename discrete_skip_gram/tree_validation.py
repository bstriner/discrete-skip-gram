import csv

import numpy as np

from .util import latest_file
from .util import write_csv, calc_utilization


def validate_encoding_tree(cooccurrence,
                           encoding,
                           z_k,
                           eps=1e-9):
    _co = cooccurrence.astype(np.float32)
    _co = _co / np.sum(_co, axis=None)
    assert encoding.ndim == 2
    z_depth = encoding.shape[1]
    x_k = encoding.shape[0]
    nlls = []
    utilizations = []
    for depth in range(z_depth):
        buckets = z_k ** (depth + 1)
        mask = np.power(z_k, np.arange(depth + 1)).reshape((1, -1))
        e = encoding[:, :(depth + 1)]
        b = np.sum(mask * e, axis=1)  # (x_k,)
        assert np.max(b) < buckets
        # nll
        m = np.zeros((buckets, x_k))  # zk, xk
        m[b, np.arange(x_k)] = 1
        p = np.dot(m, _co)  # (z_k, x_k) * (x_k, x_k) = z_k, x_k
        marg = np.sum(p, axis=1, keepdims=True)
        cond = p / (marg + eps)
        nll = np.asscalar(np.sum(p * -np.log(eps + cond), axis=None))  # scalar
        nlls.append(nll)

        # utilization
        utilizations.append(calc_utilization(b))

    return np.array(nlls), np.array(utilizations)


def write_encoding_tree(
        output_path,
        cooccurrence,
        encoding,
        z_k
):
    nlls, utilizations = validate_encoding_tree(cooccurrence=cooccurrence, encoding=encoding, z_k=z_k)
    data = [[i, n, u] for i, (n, u) in enumerate(zip(nlls, utilizations))]
    write_csv("{}.csv".format(output_path), rows=data, header=['Depth', 'NLL', 'Utilization'])
    np.savez("{}.npz".format(output_path), nlls=nlls, utilizations=utilizations)
    return nlls, utilizations


def run_tree_validation(input_path,
                        output_path,
                        cooccurrence,
                        z_k=2):
    encoding_path, epoch = latest_file(input_path, "encodings-(\d+).npy")
    if not epoch:
        raise ValueError("No file found at {}".format(input_path))
    print("Epoch {}: {}".format(epoch, encoding_path))
    encoding = np.load(encoding_path)
    return write_encoding_tree(output_path=output_path,
                               cooccurrence=cooccurrence,
                               encoding=encoding,
                               z_k=z_k)