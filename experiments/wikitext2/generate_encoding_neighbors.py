import pickle

import numpy as np
from tqdm import tqdm

from discrete_skip_gram.lm.random_encodings import random_encodings
from discrete_skip_gram.lm.write_encodings import write_encodings


def generate_neighbors(encoding):
    x_k = encoding.shape[0]
    z_k = encoding.shape[1]
    shape = (2,)*z_k
    nn = np.zeros(shape, dtype=np.int32)



    pass

def main():
    units = [16, 32, 64, 128, 256, 512]
    iters = 5

    input_path = 'output/random_encodings'
    output_path = 'output/neighbors'

    for u in tqdm(units, desc='Neighbors'):
        for i in tqdm(range(iters), desc='Units={}'.format(u)):
            ip = '{}/units-{}/iter-{}.npy'.format(input_path, u, i)
            encoding = np.load(ip)
            nn = generate_neighbors(encoding)


if __name__ == '__main__':
    main()
