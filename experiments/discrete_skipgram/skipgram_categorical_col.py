import os

import numpy as np

from discrete_skip_gram.skipgram.categorical_col import CategoricalColModel
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from keras.optimizers import Adam
from keras.regularizers import L1L2


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    opt = Adam(3e-4)
    epochs = 1000
    batches = 64
    batch_size = 8
    z_k = 1024
    regularizer = L1L2(1e-12, 1e-12)
    outputpath = "output/skipgram_categorical_col"
    type_t = 'float32'
    type_np = np.float32
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    model = CategoricalColModel(cooccurrence=cooccurrence,
                                z_k=z_k,
                                opt=opt,
                                regularizer=regularizer,
                                type_np=type_np, type_t=type_t)
    model.train(outputpath, epochs=epochs, batches=batches, batch_size=batch_size)


if __name__ == "__main__":
    main()
