import os

import numpy as np

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
from discrete_skip_gram.flat_model import FlatModel
from discrete_skip_gram.flat_validation import run_flat_validation
from discrete_skip_gram.optimizers import AdamOptimizer


def main():
    epochs = 1000
    batches = 128
    z_k = 256
    outputpath = "output/skipgram_256_m2"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    scale = 1e-1
    opt = AdamOptimizer(1e-3)
    model = FlatModel(cooccurrence=cooccurrence,
                      z_k=z_k,
                      opt=opt,
                      mode=2,
                      scale=scale)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
