from discrete_skip_gram.baseline_train import run_baseline
import numpy as np


def main():
    op = "output/skipgram_baseline"
    epochs = 20
    batches = 4096
    cooccurrence = np.load('output/cooccurrence.npy')
    z_ks = [512, 256, 128, 64, 32]
    iters = 10
    run_baseline(cooccurrence=cooccurrence,
                 z_ks=z_ks,
                 iters=iters,
                 output_path=op,
                 epochs=epochs,
                 batches=batches)


if __name__ == "__main__":
    main()
