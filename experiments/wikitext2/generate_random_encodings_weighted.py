import pickle

import numpy as np
from tqdm import tqdm

from discrete_skip_gram.lm.random_encodings import random_encodings
from discrete_skip_gram.lm.write_encodings import write_encodings


def calc_encodings(path, units, p, iters, x_k, vocab):
    for i in tqdm(range(iters), desc="Units={},P={}".format(units, p)):
        op = '{}/units-{}/p-{:.02f}/iter-{}'.format(path, units, p, i)
        e = random_encodings(x_k=x_k, p=p, units=units)
        write_encodings(output_path=op + ".csv", vocab=vocab, enc=e)
        np.save(op + ".npy", e)

def main():
    iters = 1
    with open('output/corpus/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    output_path = 'output/random_encodings_weighted'
    x_k = len(vocab)
    kwargs = {'iters':iters,'vocab':vocab, 'x_k':x_k, 'path':output_path}

    """
    units = [128]
    ps = [0.05, 0.1, 0.2, 0.5]
    for u in tqdm(units, desc='Units'):
        for p in tqdm(ps, desc='Units={}'.format(u)):
            write_encodings(units=u, p=p, **kwargs)
    """
    calc_encodings(units=1024, p=0.01, **kwargs)


if __name__ == '__main__':
    main()
