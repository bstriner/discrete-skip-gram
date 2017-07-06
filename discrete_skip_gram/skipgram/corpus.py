import os
import pickle
from collections import Counter

import numpy as np
from nltk.corpus import brown


def clean(w):
    return "".join(c for c in w.lower() if ord('a') <= ord(c) <= ord('z'))


def get_words():
    return [x for x in [clean(w) for w in brown.words()] if len(x) > 0]


def write_corpus(path, min_count=20):
    if not os.path.exists(path):
        os.makedirs(path)
    vocab_txt = os.path.join(path, 'vocab.txt')
    vocab_pkl = os.path.join(path, 'vocab.pkl')
    corpus_npy = os.path.join(path, 'corpus.npy')
    words = get_words()
    counter = Counter(words)
    vocab = [word for word, count in counter.iteritems() if count >= min_count]
    vocab.sort()
    wordmap = {w: i for i, w in enumerate(vocab)}
    n = len(words)
    data = np.zeros((n,), dtype=np.int32)
    for i, w in enumerate(words):
        if w in wordmap:
            data[i] = wordmap[w] + 1
    with open(vocab_txt, 'w') as f:
        for v in vocab:
            f.write(v)
            f.write("\n")
    with open(vocab_pkl, 'wb') as f:
        pickle.dump(vocab, f)
    np.save(corpus_npy, data)
    print "Corpus size: {}".format(n)
    print "Nonzero size: {}".format(np.count_nonzero(data))
    print "Vocab size: {}".format(len(vocab))


def load_corpus(path):
    vocab_pkl = os.path.join(path, 'vocab.pkl')
    corpus_npy = os.path.join(path, 'corpus.npy')
    with open(vocab_pkl, 'rb') as f:
        vocab = pickle.load(f)
    corpus = np.load(corpus_npy)
    return vocab, corpus
