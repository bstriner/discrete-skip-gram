import os

import numpy as np
from discrete_skip_gram.skipgram.discrete_full_acc import DiscreteFullAccModel
from discrete_skip_gram.skipgram.optimizers import AdamOptimizer
from discrete_skip_gram.skipgram.tree_parameterization import ParameterizationBU
from keras.regularizers import L1L2

from discrete_skip_gram.cooccurrence import load_cooccurrence

"""
Fully parameterized (each branch is not independent)
descent performed over full covariance matrix
"""


def main():
    batch_size = 8
    opt = AdamOptimizer(1e-3)
    regularizer = L1L2(1e-11, 1e-11)
    outputpath = "output/skipgram_discrete_bu_acc"
    z_depth = 10
    z_k = 2
    epochs = 1000
    batches = 64
    type_t = 'float32'
    type_np = np.float32
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    schedule = np.power(1.8, np.arange(z_depth))
    schedule = schedule / np.max(schedule)
    model = DiscreteFullAccModel(cooccurrence=cooccurrence,
                                 z_k=z_k,
                                 opt=opt,
                                 regularizer=regularizer,
                                 z_depth=z_depth,
                                 schedule=schedule,
                                 param_class=ParameterizationBU,
                                 type_np=type_np,
                                 type_t=type_t)
    model.train(outputpath=outputpath,
                epochs=epochs,
                batches=batches,
                batch_size=batch_size)


if __name__ == "__main__":
    main()
