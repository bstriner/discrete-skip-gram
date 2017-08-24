import numpy as np

from discrete_skip_gram.point_replace import train_battery


def main():
    cooccurrence = np.load('output/cooccurrence.npy')
    z_k = 2
    train_battery(output_path="output/point_replace_binary",
                  cooccurrence=cooccurrence,
                  z_k=z_k,
                  iters=10)


if __name__ == '__main__':
    main()
