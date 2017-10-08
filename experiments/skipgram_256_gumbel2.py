import os

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np
from keras.optimizers import Adam

from discrete_skip_gram.flat_validation import run_flat_validation
from discrete_skip_gram.gumbel_model2 import GumbelModel2
from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.regularizers import EntropyRegularizer


def main():
    # hyperparameters
    epochs = 1000
    batches = 4096
    z_k = 256
    initializer = uniform_initializer(0.05)
    scale = 0.05
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    initial_pz_weight = np.random.uniform(low=-scale, high=scale, size=(cooccurrence.shape[0], z_k))
    opt = Adam(1e-3)
    decay = 1e-4
    pz_regularizer = EntropyRegularizer(3e-6)
    # build and train
    outputpath = "output/skipgram_256_gumbel2"
    model = GumbelModel2(cooccurrence=cooccurrence,
                         z_k=z_k,
                         opt=opt,
                         initializer=initializer,
                         initial_pz_weight=initial_pz_weight,
                         pz_regularizer=pz_regularizer,
                         decay=decay)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
