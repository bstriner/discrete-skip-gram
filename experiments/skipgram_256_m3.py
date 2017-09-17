import os

import numpy as np

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
from discrete_skip_gram.flat_model import FlatModel
from discrete_skip_gram.flat_validation import run_flat_validation
from keras.optimizers import Adam
from discrete_skip_gram.regularizers import BalanceRegularizer, ExclusiveLasso

def main():
    epochs = 1000
    batches = 4096
    z_k = 256
    outputpath = "output/skipgram_256_m3"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    scale = 1e-2
    opt = Adam(1e-3)
    pz_regularizer = BalanceRegularizer(1e-8)
    pz_weight_regularizer = ExclusiveLasso(1e-12)
    model = FlatModel(cooccurrence=cooccurrence,
                      z_k=z_k,
                      opt=opt,
                      mode=3,
                      pz_regularizer=pz_regularizer,
                      pz_weight_regularizer=pz_weight_regularizer,
                      scale=scale)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
