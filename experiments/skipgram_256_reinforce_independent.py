import os

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np
from keras.optimizers import Adam
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_skip_gram.flat_validation import run_flat_validation
from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.reinforce_model import ReinforceModel
from discrete_skip_gram.reinforce_parameterizations.independent_parameterization import IndependentParameterization


def main():
    # hyperparameters
    epochs = 1000
    batches = 4096
    z_k = 256
    initializer = uniform_initializer(5)
    srng = RandomStreams(123)
    opt = Adam(1e-3)
    # build and train
    outputpath = "output/skipgram_256_reinforce_independent"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    parameterization = IndependentParameterization(x_k=cooccurrence.shape[0],
                                                   z_k=z_k,
                                                   initializer=initializer,
                                                   srng=srng)
    model = ReinforceModel(cooccurrence=cooccurrence,
                           parameterization=parameterization,
                           z_k=z_k,
                           opt=opt)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
