import os

import numpy as np

from discrete_skip_gram.skipgram.categorical_col_acc import CategoricalColAccModel
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.optimizers import AdamOptimizer
from keras.regularizers import L1L2


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    opt = AdamOptimizer(lr=1e-3)
    epochs = 1000
    batches = 64
    batch_size = 8
    z_k = 1024
    regularizer = L1L2(1e-12, 1e-12)
    outputpath = "output/skipgram_categorical_col_acc"
    type_t = 'float32'
    type_np = np.float32
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    model = CategoricalColAccModel(cooccurrence=cooccurrence,
                                   z_k=z_k,
                                   opt=opt,
                                   # regularizer=regularizer,
                                   type_np=type_np, type_t=type_t)
    print "Training"
    model.train(outputpath, epochs=epochs, batches=batches, batch_size=batch_size)


if __name__ == "__main__":
    main()
