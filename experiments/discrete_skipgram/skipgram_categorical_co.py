import csv
import itertools
import os

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.tensor_util import softmax_nd
from keras.optimizers import Adam
from keras.regularizers import L1L2

from discrete_skip_gram.skipgram.categorical import CategoricalModel

def main():
    batch_size = 64
    opt = Adam(3e-4)
    regularizer = L1L2(1e-13, 1e-13)
    outputpath = "output/skipgram_discrete_categorical"
    z_k = 1024
    epochs = 1000
    batches = 64

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    model = CategoricalModel(cooccurrence=cooccurrence, z_k=z_k, opt=opt, regularizer=regularizer)
    model.train(outputpath=outputpath, epochs=epochs, batches=batches, batch_size=batch_size)

if __name__ == "__main__":
    main()
