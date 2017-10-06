import os

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np
from keras.optimizers import Adam
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_skip_gram.flat_validation import run_flat_validation
from discrete_skip_gram.reinforce_model import ReinforceModel
from discrete_skip_gram.reinforce_parameterizations.independent_parameterization import IndependentParameterization
from discrete_skip_gram.reinforce_train import calc_initial_pz
from discrete_skip_gram.regularizers import EntropyRegularizer

def main():
    epochs = 1000
    batches = 4096
    z_k = 256
    opt = Adam(1e-3)
    smoothing = 0.1
    noise = 1

    inputpath = 'output/skipgram_256-b/b-1.0e-04/iter-0/encodings-00000009.npy'
    outputpath = "output/skipgram_256_finetune_reinforce_independent"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    pz_regularizer = EntropyRegularizer(1e-6)
    # create initial weights
    encoding = np.load(inputpath)
    initial_pz = calc_initial_pz(encoding=encoding, z_k=z_k, smoothing=smoothing, noise=noise)

    srng = RandomStreams(123)
    parameterization = IndependentParameterization(x_k=cooccurrence.shape[0],
                                                   z_k=z_k,
                                                   initializer=None,
                                                   initial_pz_weight=initial_pz,
                                                   pz_regularizer=pz_regularizer,
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
