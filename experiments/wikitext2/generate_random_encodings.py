import pickle

import numpy as np
from tqdm import tqdm

from discrete_skip_gram.lm.random_encodings import random_encodings
from discrete_skip_gram.lm.write_encodings import write_encodings


def main():
    units = [16, 32, 64, 128, 256, 512]
    iters = 5
    p = 0.5
    output_path = 'output/random_encodings'

    with open('output/corpus/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    x_k = len(vocab)

    for u in tqdm(units, desc='Units'):
        for i in tqdm(range(iters), desc="Units={}".format(u)):
            op = '{}/units-{}/iter-{}'.format(output_path, u, i)
            e = random_encodings(x_k=x_k, p=p, units=u)
            write_encodings(output_path=op + ".csv", vocab=vocab, enc=e)
            np.save(op + ".npy", e)


if __name__ == '__main__':
    main()
