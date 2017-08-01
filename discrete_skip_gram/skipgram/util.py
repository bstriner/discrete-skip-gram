import csv
import itertools
import math
import os
import re
from os import listdir
from os.path import join

import numpy as np


def make_path(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def latest_file(path, fmt):
    prog = re.compile(fmt)
    latest_epoch = -1
    latest_m = None
    for f in listdir(path):
        m = prog.match(f)
        if m:
            epoch = int(m.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_m = f
    if latest_m:
        return join(path, latest_m), latest_epoch
    else:
        return None, None


def generate_batches(data, batch_size):
    n = data.shape[0]
    batch_count = int(math.ceil(float(n) / float(batch_size)))
    batches = []
    for i in range(batch_count):
        idx0 = batch_size * i
        idx1 = batch_size * (i + 1)
        if idx1 > n:
            idx1 = n
        batch = data[idx0:idx1]
        batches.append(batch)
    return batches


def generate_sequences(z_depth, z_k):
    data = np.array(list(itertools.product(list(range(z_k)), repeat=z_depth + 1))).astype(np.int32)
    return data


def array_string(array, fmt="{:.3f}", cat=", "):
    return cat.join(fmt.format(np.asscalar(d)) for d in array)


def write_encodings(path, vocab, encodings):
    x_k = encodings.shape[0]
    z_k = encodings.shape[1]
    make_path(path)
    with open(path, 'wb') as f:
        w = csv.writer(f)
        w.writerow(["Id", "Word", "Encoding"])
        for i in range(x_k):
            if i == 0:
                word = "_UNK_"
            else:
                word = vocab[i - 1]
            enc = encodings[i, :]
            encs = "".join(chr(ord('a') + enc[j]) for j in range(enc.shape[0]))
            w.writerow([i, word, encs])

def write_csv(path, rows, header=None):
    make_path(path)
    with open(path, 'wb') as f:
        w= csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)
