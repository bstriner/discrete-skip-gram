import numpy as np

from discrete_skip_gram.point_replace import PointReplaceModel


def main():
    cooccurrence = np.load('output/cooccurrence.npy')
    z_k = 2
    m = PointReplaceModel(cooccurrence=cooccurrence, z_k=z_k)
    m.train(output_path='output/point_replace_binary')


if __name__ == '__main__':
    main()
