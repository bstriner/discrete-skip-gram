import os

import numpy as np
from keras.optimizers import Adam

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
from discrete_skip_gram.flat_model import FlatModel
from discrete_skip_gram.flat_validation import run_flat_validation


def main():
    epochs = 1000
    batches = 4096
    z_k = 1024
    outputpath = "output/skipgram_flat_tst2"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    scale = 1e-2
    opt = Adam(1e-3)
    model = FlatModel(cooccurrence=cooccurrence,
                      z_k=z_k,
                      opt=opt,
                      scale=scale)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
