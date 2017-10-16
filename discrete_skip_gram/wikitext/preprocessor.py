import csv
import itertools
import os
import pickle

import numpy as np

from .tokenizer import tokenize
from ..util import make_dir


def preprocess(data_path, output_path):
    files = ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']
    paths = [os.path.join(data_path, f) for f in files]
    tokens = [tokenize(path) for path in paths]
    vocab = list(set(itertools.chain.from_iterable(tokens)))
    vocab.sort()
    vocab_map = {v: i for i, v in enumerate(vocab)}
    corpus = [np.array([vocab_map[t] for t in tok], dtype=np.int32) for tok in tokens]
    make_dir(output_path)
    with open(os.path.join(output_path, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    np.savez(os.path.join(output_path, 'corpus.npz'),
             train=corpus[0],
             valid=corpus[1],
             test=corpus[2])
    with open(os.path.join(output_path, 'vocab.csv'), 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Token'])
        for i, v in enumerate(vocab):
            w.writerow([i, v])
    with open(os.path.join(output_path, 'summary.txt'), 'w') as f:
        f.write("Train tokens: {}\n".format(len(tokens[0])))
        f.write("Validation tokens: {}\n".format(len(tokens[1])))
        f.write("Test tokens: {}\n".format(len(tokens[2])))
        f.write("Unique Vocabulary: {}\n".format(len(vocab)))
        f.write("Unique Vocabulary (ignore case): {}\n".format(len(set(v.lower() for v in vocab))))
