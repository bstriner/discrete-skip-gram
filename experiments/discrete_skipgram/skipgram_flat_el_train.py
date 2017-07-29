import os

import numpy as np

from discrete_skip_gram.skipgram.categorical_col import CategoricalColModel
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from keras.optimizers import Adam
from discrete_skip_gram.skipgram.regularizers import ExclusiveLasso

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    opt = Adam(1e-3)
    epochs = 1000
    batches = 1024
    z_k = 1024
    regularizer = ExclusiveLasso(1e-2)
    outputpath = "output/skipgram_flat-el"
    type_t = 'float32'
    type_np = np.float32
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    model = CategoricalColModel(cooccurrence=cooccurrence,
                                z_k=z_k,
                                opt=opt,
                                pz_weight_regularizer=regularizer,
                                type_np=type_np, type_t=type_t)
    model.train(outputpath, epochs=epochs, batches=batches)


if __name__ == "__main__":
    main()
