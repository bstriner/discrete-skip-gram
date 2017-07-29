import csv

import numpy as np

from .cooccurrence import load_cooccurrence


def validate_depth(depth, encoding, co, z_k):
    count = z_k ** (depth + 1)
    enc = encoding[:, :depth + 1]
    mask = np.reshape(np.power(z_k, np.arange(depth + 1)), (1, -1))
    buckets = np.sum(mask * enc, axis=1)
    n = np.sum(co, axis=None)
    eps = 1e-9
    tot = 0.
    for bucket in range(count):
        ind = np.nonzero(np.equal(buckets, bucket))[0]
        if ind.shape[0] == 0:
            # pass
            print "Empty tree: {}".format(bucket)
        else:
            rows = co[ind, :]
            d = np.sum(rows, axis=0)
            p = d / np.sum(d)
            nll = np.sum(p * -np.log(p + eps))
            mp = np.sum(d) / n
            tot += nll * mp
    return tot


def calc_validation(encoding, co, z_k):
    z_depth = encoding.shape[1]
    nlls = []
    for depth in range(z_depth):
        nll = validate_depth(depth=depth, encoding=encoding, co=co, z_k=z_k)
        nlls.append(nll)
        print "NLL {}: {}".format(depth, nll)
    return nlls


def write_validation(output_path, encoding, co, z_k):
    val = calc_validation(encoding=encoding, co=co, z_k=z_k)
    with open(output_path, 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Bits', 'NLL'])
        for i, v in enumerate(val):
            w.writerow([i + 1, v])
        vals = np.array(val)
        sched = np.power(1.5, np.arange(vals.shape[0]))
        loss = np.sum(vals * sched)
        w.writerow(['Total', loss])
        print("Loss: {}".format(loss))


def validate_binary(output_path, cooccurrence_path, encoding_path):
    x = load_cooccurrence(cooccurrence_path)
    encoding = np.load(encoding_path)
    write_validation(output_path=output_path, encoding=encoding, co=x, z_k=2)


def validate_flat(output_path, cooccurrence_path, encoding_path, z_k):
    x = load_cooccurrence(cooccurrence_path)
    encoding = np.load(encoding_path)
    encoding = np.expand_dims(encoding, axis=1)  # (x_k, 1)
    print "Encoding {}".format(encoding.shape)
    write_validation(output_path=output_path, encoding=encoding, co=x, z_k=z_k)


def validate_encoding_flat(cooccurrence, eps=1e-9):
    _co = cooccurrence.astype(np.float32)
    _co = _co / np.sum(_co, axis=None)

    def fun(enc, z_k):
        x_k = _co.shape[0]
        m = np.zeros((z_k, x_k))  # zk, xk
        m[enc, np.arange(x_k)] = 1
        p = np.dot(m, _co)  # (z_k, x_k) * (x_k, x_k) = z_k, x_k
        marg = np.sum(p, axis=1, keepdims=True)
        cond = p / (marg + eps)
        nll = np.sum(cond * -np.log(eps + cond), axis=1, keepdims=True)  # (z_k, 1)
        loss = np.asscalar(np.sum(nll * marg, axis=None))
        return loss

    return fun


def validate_empty(enc, z_k):
    assert len(enc.shape) == 1
    used = set(enc[i] for i in range(enc.shape[0]))
    print "Used: {}/{}".format(len(used), z_k)
