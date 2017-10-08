import os

import keras.backend as K
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np
from keras.optimizers import Adam

from discrete_skip_gram.flat_validation import run_flat_validation
from discrete_skip_gram.gumbel_model3 import GumbelModel3
from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.mlp import MLP
from discrete_skip_gram.regularizers import EntropyRegularizer
from discrete_skip_gram.tensor_util import leaky_relu


def main():
    # hyperparameters
    epochs = 1000
    batches = 2048
    z_k = 256
    initializer = uniform_initializer(0.1)
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    opt = Adam(1e-3)
    units = 512
    tao0 = 5.
    tao_min = 0.1
    tao_decay = 2e-4
    mlp = MLP(input_units=z_k,
              output_units=cooccurrence.shape[0],
              hidden_units=units,
              hidden_depth=1,
              initializer=initializer,
              use_bn=True,
              hidden_activation=leaky_relu)
    initial_b = np.log(np.sum(cooccurrence, axis=0))
    K.set_value(mlp.bs[-1], initial_b)
    pz_regularizer = EntropyRegularizer(3e-7)
    # build and train
    outputpath = "output/skipgram_256_gumbel3"
    model = GumbelModel3(cooccurrence=cooccurrence,
                         z_k=z_k,
                         opt=opt,
                         mlp=mlp,
                         initializer=initializer,
                         pz_regularizer=pz_regularizer,
                         tao0=tao0,
                         tao_min=tao_min,
                         tao_decay=tao_decay)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
