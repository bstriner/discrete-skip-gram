import os

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np
from keras.optimizers import Adam

from discrete_skip_gram.flat_validation import run_flat_validation
from discrete_skip_gram.gumbel_model1 import GumbelModel1
from discrete_skip_gram.initializers import uniform_initializer


def main():
    # hyperparameters
    epochs = 1000
    batches = 4096
    z_k = 256
    initializer = uniform_initializer(0.05)
    opt = Adam(1e-3)
    decay = 1e-6
    # build and train
    outputpath = "output/skipgram_256_gumbel1"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    model = GumbelModel1(cooccurrence=cooccurrence,
                         z_k=z_k,
                         opt=opt,
                         initializer=initializer,
                         decay=decay)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
