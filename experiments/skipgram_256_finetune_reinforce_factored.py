import os

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np
from keras.optimizers import Adam

from discrete_skip_gram.flat_validation import run_flat_validation
from discrete_skip_gram.regularizers import EntropyRegularizer
from discrete_skip_gram.reinforce_factored_model import ReinforceFactoredModel
from discrete_skip_gram.reinforce_train import calc_initial_pz


def main():
    epochs = 1000
    batches = 4096
    z_k = 256
    smoothing = 0.2
    noise = 1.
    opt = Adam(1e-3)
    pz_regularizer = EntropyRegularizer(1e-5)

    inputpath = 'output/skipgram_256-b/b-1.0e-04/iter-0/encodings-00000009.npy'
    outputpath = "output/skipgram_256_finetune_reinforce_factored"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)

    # create initial weights
    encoding = np.load(inputpath)
    initial_pz = calc_initial_pz(encoding=encoding, z_k=z_k, smoothing=smoothing, noise=noise)
    model = ReinforceFactoredModel(cooccurrence=cooccurrence,
                                   z_k=z_k,
                                   opt=opt,
                                   initial_pz_weight=initial_pz,
                                   pz_regularizer=pz_regularizer,
                                   initializer=None)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
