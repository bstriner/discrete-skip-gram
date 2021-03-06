import numpy as np

from discrete_skip_gram.regularizers import ExclusiveLasso
from discrete_skip_gram.tree_train import train_tree_regularizer_battery


def main():
    epochs = 10
    iters = 1
    batches = 4096
    z_k = 2
    z_depth = 10
    outputpath = "output/skipgram_tree_el"
    betas = [
        0.5,
        0.85,
        1.2,
        2.0
    ]
    weights = [1e-13,
               1e-12,
               1e-11,
               1e-10,
               1e-9,
               1e-8]
    labels = ["el-{:.01e}".format(w) for w in weights]
    regularizers = [ExclusiveLasso(w) for w in weights]
    train_tree_regularizer_battery(
        betas=betas,
        epochs=epochs,
        iters=iters,
        batches=batches,
        z_k=z_k,
        z_depth=z_depth,
        outputpath=outputpath,
        labels=labels,
        is_weight_regularizer=True,
        kwdata={'weights': np.array(weights)},
        regularizers=regularizers)


if __name__ == "__main__":
    main()
