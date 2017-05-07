import itertools

import numpy as np

from dataset import load_dataset
from discrete_skip_gram.hsm import HSM
import os
import csv
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

def write_hsm(dataset, codes, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    n = codes.shape[0]
    m = codes.shape[1]
    with open(output_path, 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Id','Word','Encoding']+['Cat {}'.format(i) for i in range(m)])
        for i in range(n):
            word = dataset.get_word(i)
            encs = [codes[i,j] for j in range(m)]
            encf = "".join(chr(ord('a')+e) for e in encs)
            w.writerow([i, word, encf]+encs)

hsm_path = "output/brown/random_hsm.pkl"


def load_hsm():
    return HSM.load(hsm_path)

def build_and_write(dataset):
    print "building"
    hsm = build_hsm(dataset.k)
    print "saving"
    hsm.save(hsm_path)

def load_and_test_hsm(dataset):
    hsm = load_hsm()
    write_hsm(dataset, hsm.codes, "output/brown/random_hsm.csv")
    for i in range(10):
        w = np.random.randint(0, dataset.k)
        word = dataset.get_word(w)
        code = hsm.encode(w)
        w2 = hsm.decode(code)
        print "Word {}: {}".format(w, word)
        print "Code: {}, Decoded: {}, {}".format(code, w2, dataset.get_word(w2))

def main():
    dataset = load_dataset()
    #build_and_write(dataset)
    load_and_test_hsm(dataset)

if __name__ == "__main__":
    main()
