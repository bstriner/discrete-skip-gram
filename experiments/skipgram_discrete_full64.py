import os

import numpy as np
from discrete_skip_gram.skipgram.discrete_full import DiscreteFullModel
from keras.optimizers import Adam
from keras.regularizers import L1L2

from discrete_skip_gram.cooccurrence import load_cooccurrence


def main():
    batch_size = 64
    opt = Adam(3e-4)
    regularizer = L1L2(1e-11, 1e-11)
    outputpath = "output/skipgram_discrete_full64"
    z_depth = 10
    z_k = 2
    epochs = 1000
    batches = 64
    type_t = 'float64'
    type_np = np.float64
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy')
    schedule = np.power(2., np.arange(z_depth))
    model = DiscreteFullModel(cooccurrence=cooccurrence, z_k=z_k, opt=opt, regularizer=regularizer,
                              z_depth=z_depth, schedule=schedule,
                              type_np=type_np, type_t=type_t)
    model.train(outputpath=outputpath, epochs=epochs, batches=batches, batch_size=batch_size)


if __name__ == "__main__":
    main()
