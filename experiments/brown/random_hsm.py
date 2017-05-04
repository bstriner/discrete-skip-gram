import itertools

import numpy as np

from dataset import load_dataset
from discrete_skip_gram.hsm import HSM


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
    return HSM(codes, words)


hsm_path = "output/brown/random_hsm.pkl"


def load_hsm():
    return HSM.load(hsm_path)


def main():
    print "loading"
    dataset = load_dataset()
    print "building"
    hsm = build_hsm(dataset.k)
    print "saving"
    hsm.save(hsm_path)
    for i in range(10):
        w = np.random.randint(0, dataset.k)
        word = dataset.get_word(w)
        code = hsm.encode(w)
        w2 = hsm.decode(code)
        print "Word {}: {}".format(w, word)
        print "Code: {}, Decoded: {}, {}".format(code, w2, dataset.get_word(w2))


if __name__ == "__main__":
    main()
