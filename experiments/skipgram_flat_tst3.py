import os

import numpy as np
from keras.optimizers import Adam, SGD, RMSprop

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
from discrete_skip_gram.flat_model import FlatModel
from discrete_skip_gram.flat_validation import run_flat_validation
from discrete_skip_gram.regularizers import ExclusiveLasso


def main():
    epochs = 1000
    batches = 4096
    z_k = 1024
    outputpath = "output/skipgram_flat_tst3"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    zpath = "output/z-00000016.npy"
    enc = np.load(zpath)
    reg = ExclusiveLasso(1e-10)
    z = np.zeros((cooccurrence.shape[0], z_k), dtype='float32')
    z[np.arange(z.shape[0]), enc] = 1
    pz = np.dot(np.transpose(z, (1,0)), cooccurrence/np.sum(cooccurrence, axis=None))
    m = np.sum(pz, axis=1, keepdims=True)
    c = pz/m
    h = np.sum(pz * -np.log(1e-9+c), axis=None)
    print("H: {}".format(h))
    scale = 1e-1
    initial_pz = z * scale
    opt = SGD(1e4)
    model = FlatModel(cooccurrence=cooccurrence,
                      z_k=z_k,
                      pz_weight_regularizer=reg,
                      opt=opt,
                      initial_pz=initial_pz,
                      scale=scale)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
