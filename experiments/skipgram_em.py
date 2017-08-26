import numpy as np

from discrete_skip_gram.em import train_battery


def main():
    cooccurrence = np.load('output/cooccurrence.npy')
    z_k = 1024
    train_battery(output_path="output/em_1024",
                  cooccurrence=cooccurrence,
                  z_k=z_k,
                  iters=10)


if __name__ == '__main__':
    main()
