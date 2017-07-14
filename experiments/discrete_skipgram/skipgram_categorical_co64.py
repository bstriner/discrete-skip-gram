import os

import numpy as np

from discrete_skip_gram.skipgram.categorical import CategoricalModel
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from keras.optimizers import Adam
from keras.regularizers import L1L2


def main():
    batch_size = 64
    opt = Adam(3e-4)
    regularizer = L1L2(1e-12, 1e-12)
    outputpath = "output/skipgram_categorical_co64"
    z_k = 1024
    epochs = 1000
    batches = 128
    type_t = 'float64'
    type_np = np.float64
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy')
    model = CategoricalModel(cooccurrence=cooccurrence, z_k=z_k, opt=opt, regularizer=regularizer,
                             type_np=type_np, type_t=type_t)
    model.train(outputpath=outputpath, epochs=epochs, batches=batches, batch_size=batch_size)


if __name__ == "__main__":
    main()
