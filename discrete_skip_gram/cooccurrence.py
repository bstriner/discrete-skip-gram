import numpy as np

from .corpus import load_corpus
from .dataset import load_dataset
from .util import make_path


def get_cooccurrence(dataset_path, corpus_path):
    vocab, corpus = load_corpus(corpus_path)
    dataset = load_dataset(dataset_path)
    n = dataset.shape[0]
    m = len(vocab) + 1
    x = np.zeros((m, m), dtype=np.float32)
    for i in range(n):
        a = dataset[i, 0]
        b = dataset[i, 1]
        x[a, b] += 1
        x[b, a] += 1
    return x


def write_cooccurrence(cooccurrence_path, dataset_path, corpus_path):
    make_path(cooccurrence_path)
    x = get_cooccurrence(dataset_path, corpus_path)
    np.save(cooccurrence_path, x)


def load_cooccurrence(path):
    return np.load(path)
