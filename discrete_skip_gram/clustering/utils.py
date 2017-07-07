import csv
import os
import numpy as np


def write_encodings(enc, vocab, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    np.save(output_path + ".npy", enc)
    depth = enc.shape[1]
    with open(output_path + ".csv", 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Idx', 'Word', 'Encoding'] + ['Cat {}'.format(i) for i in range(depth)])
        for i in range(enc.shape[0]):
            e = enc[i, :]
            ea = [e[j] for j in range(e.shape[0])]
            ef = "".join(chr(ord('a') + x) for x in ea)
            if i > 0:
                word = vocab[i-1]
            else:
                word = "_UNK_"
            w.writerow([i, word, ef] + ea)
