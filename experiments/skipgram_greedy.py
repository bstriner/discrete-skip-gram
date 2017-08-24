import numpy as np

from discrete_skip_gram.greedy import GreedyModel


def main():
    cooccurrence = np.load('output/cooccurrence.npy')
    z_k = 1024
    repeats = 1
    m = GreedyModel(cooccurrence=cooccurrence, z_k=z_k, repeats=repeats)
    m.train(output_path='output/greedy')


if __name__ == '__main__':
    main()
