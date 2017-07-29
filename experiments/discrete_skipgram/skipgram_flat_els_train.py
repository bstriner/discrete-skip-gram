import os

import numpy as np

from discrete_skip_gram.skipgram.categorical_col import CategoricalColModel
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from keras.optimizers import Adam
from discrete_skip_gram.skipgram.regularizers import ExclusiveLassoSqrt

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    opt = Adam(1e-3)
    epochs = 1000
    batches = 1024
    z_k = 1024
    regularizer = ExclusiveLassoSqrt(1e-6)
    outputpath = "output/skipgram_flat-els"
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    model = CategoricalColModel(cooccurrence=cooccurrence,
                                z_k=z_k,
                                opt=opt,
                                pz_weight_regularizer=regularizer)
    model.train(outputpath, epochs=epochs, batches=batches)


if __name__ == "__main__":
    main()
