import numpy as np
from discrete_skip_gram.skipgram.tree_train import train_regularizer_battery

from discrete_skip_gram.regularizers import BalanceRegularizer


def main():
    epochs = 10
    iters = 1
    batches = 4096
    z_k = 2
    z_depth = 10
    outputpath = "output/skipgram_tree_b"
    betas = [0.85,
             1.2]
    weights = [1e-3,
               1e-4,
               1e-5]
    labels = ["{:.01e}".format(w) for w in weights]
    regularizers = [BalanceRegularizer(w) for w in weights]
    train_regularizer_battery(
        betas=betas,
        epochs=epochs,
        iters=iters,
        batches=batches,
        z_k=z_k,
        z_depth=z_depth,
        outputpath=outputpath,
        labels=labels,
        is_weight_regularizer=False,
        kwdata={'weights': np.array(weights)},
        regularizers=regularizers)


if __name__ == "__main__":
    main()
