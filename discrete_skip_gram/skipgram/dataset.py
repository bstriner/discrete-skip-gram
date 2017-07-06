import numpy as np

from .corpus import load_corpus
from .util import make_path


def generate_skipgrams(corpus_path, window=3):
    vocab, corpus = load_corpus(corpus_path)
    arrays = []
    for d in range(1, window + 1):
        print 'D: {}'.format(d)
        a = corpus[:-d]
        b = corpus[d:]
        grams = np.stack((a, b), axis=1)
        arrays.append(grams)
    skipgrams = np.concatenate(arrays, axis=0)
    return skipgrams


def write_dataset(dataset_path, corpus_path, window=3):
    make_path(dataset_path)
    dataset = generate_skipgrams(corpus_path=corpus_path, window=window)
    np.save(dataset_path, dataset)
    print "Dataset: {}".format(dataset.shape)


def load_dataset(dataset_path):
    return np.load(dataset_path)
