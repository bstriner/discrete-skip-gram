import os

import numpy as np

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.discrete_full_acc import DiscreteFullAccModel
from discrete_skip_gram.skipgram.optimizers import AdamOptimizer
from keras.regularizers import L1L2

"""
Fully parameterized (each branch is not independent)
descent performed over full covariance matrix
"""


def main():
    batch_size = 8
    opt = AdamOptimizer(3e-4)
    regularizer = L1L2(1e-12, 1e-12)
    outputpath = "output/skipgram_discrete_full_acc"
    z_depth = 10
    z_k = 2
    epochs = 1000
    batches = 64
    type_t = 'float32'
    type_np = np.float32
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    schedule = np.power(1.5, np.arange(z_depth))
    model = DiscreteFullAccModel(cooccurrence=cooccurrence,
                                 z_k=z_k,
                                 opt=opt,
                                 regularizer=regularizer,
                                 z_depth=z_depth, schedule=schedule,
                                 type_np=type_np, type_t=type_t)
    model.train(outputpath=outputpath, epochs=epochs, batches=batches, batch_size=batch_size)


if __name__ == "__main__":
    main()
