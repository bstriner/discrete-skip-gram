import itertools
import os
import pickle

import numpy as np

from dataset import load_dataset


def build_hsm(k):
    words = np.arange(k, dtype=np.int32)
    np.random.shuffle(words)
    depth = int(np.ceil(np.log(k) / np.log(2)))
    buckets = int(np.power(2, depth))
    print "K: {}".format(k)
    print "Depth: {:d}, Buckets: {:d}".format(depth, buckets)
    codes = np.zeros((k, depth), dtype=np.int32)
    for i, code in enumerate(itertools.product([0, 1], repeat=depth)):
        if i < words.shape[0]:
            word = words[i]
            codes[word, :] = code
    return codes, words


hsm_path = "output/brown/random_hsm.pkl"


def save_hsm(codes, words):
    if os.path.exists(hsm_path):
        raise ValueError("Path already exists: {}".format(hsm_path))
    if not os.path.exists(os.path.dirname(hsm_path)):
        os.makedirs(os.path.dirname(hsm_path))
    with open(hsm_path, 'wb') as f:
        pickle.dump((codes, words), f)


def load_hsm(hsm):
    with open(hsm_path, 'rb') as f:
        return pickle.load(f)


def decode(code, words):
    d = code.shape[0]
    i = np.sum(np.power(2, np.arange(d)[::-1]) * code)
    if i < words.shape[0]:
        return words[i]
    else:
        return 0


def main():
    print "loading"
    dataset = load_dataset()
    print "building"
    codes, words = build_hsm(dataset.k)
    print "saving"
    save_hsm(codes, words)
    for i in range(10):
        w = np.random.randint(0, dataset.k)
        word = dataset.get_word(w)
        code = codes[w]
        w2 = decode(code, words)
        print "Word {}: {}".format(w, word)
        print "Code: {}, Decoded: {}, {}".format(code, w2, dataset.get_word(w2))


if __name__ == "__main__":
    main()
