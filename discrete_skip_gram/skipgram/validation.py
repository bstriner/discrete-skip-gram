import csv

import numpy as np

from .cooccurrence import load_cooccurrence


def validate_depth(depth, encoding, co):
    count = 2 ** (depth + 1)
    enc = encoding[:, :depth + 1]
    mask = np.reshape(np.power(2, np.arange(depth + 1)), (1, -1))
    buckets = np.sum(mask * enc, axis=1)
    n = np.sum(co, axis=None)
    eps = 1e-9
    tot = 0.
    for bucket in range(count):
        ind = np.nonzero(np.equal(buckets, bucket))[0]
        if ind.shape[0] == 0:
            pass
            # print "Empty tree: {}".format(bucket)
        else:
            rows = co[ind, :]
            d = np.sum(rows, axis=0)
            p = d / np.sum(d)
            nll = np.sum(p * -np.log(p + eps))
            mp = np.sum(d) / n
            tot += nll * mp
    return tot


def calc_validation(encoding, co):
    z_depth = encoding.shape[1]
    nlls = []
    for depth in range(z_depth):
        nll = validate_depth(depth, encoding, co)
        nlls.append(nll)
        print "NLL {}: {}".format(depth, nll)
    return nlls


def write_validation(output_path, encoding, co):
    val = calc_validation(encoding, co)
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


def validate(output_path, cooccurrence_path, encoding_path):
    x = load_cooccurrence(cooccurrence_path)
    encoding = np.load(encoding_path)
    write_validation(output_path=output_path, encoding=encoding, co=x)
