import os

import numpy as np
from keras.optimizers import Adam
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_skip_gram.flat_validation import run_flat_validation
from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.reinforce_model import ReinforceModel
from discrete_skip_gram.reinforce_parameterizations.independent_parameterization import IndependentParameterization
from discrete_skip_gram.util import one_hot_np


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    epochs = 1000
    batches = 4096
    z_k = 256
    opt = Adam(1e-3)
    initializer = uniform_initializer(0.05)
    smoothing = 0.1

    inputpath = 'output/skipgram_256-b/b-1.0e-04/iter-0/encodings-00000009.npy'
    outputpath = "output/skipgram_256_finetune_reinforce_independent"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)

    # create initial weights
    enc = np.load(inputpath)
    w = one_hot_np(enc, k=z_k)
    w = (w * (1. - smoothing)) + (smoothing / z_k)
    w = np.log(w)
    w -= np.max(w, axis=1, keepdims=True)

    srng = RandomStreams(123)
    parameterization = IndependentParameterization(x_k=cooccurrence.shape[0],
                                                   z_k=z_k,
                                                   initializer=initializer,
                                                   initial_pz_weight=w,
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
