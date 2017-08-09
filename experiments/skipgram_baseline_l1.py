from discrete_skip_gram.baseline_train import run_baseline
from keras.regularizers import l1
import numpy as np


def main():
    op = "output/skipgram_baseline-l1"
    epochs = 20
    batches = 4096
    cooccurrence = np.load('output/cooccurrence.npy')
    z_ks = [512, 256, 128, 64, 32]
    iters = 3
    run_baseline(cooccurrence=cooccurrence,
                 z_ks=z_ks,
                 iters=iters,
                 output_path=op,
                 epochs=epochs,
                 batches=batches,
                 regularizer=l1(1e-8))


if __name__ == "__main__":
    main()
