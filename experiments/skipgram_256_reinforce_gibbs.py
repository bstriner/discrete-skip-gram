#import os
#os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np
from keras.optimizers import Adam

from discrete_skip_gram.reinforce_gibbs_model import ReinforceGibbsModel


def main():
    epochs = 1000
    batches = 4096
    z_k = 256
    steps = 5
    gibbs_batch_size = 256
    outputpath = "output/skipgram_256_reinforce_gibbs"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    scale = 1e-2
    opt = Adam(1e-3)
    model = ReinforceGibbsModel(cooccurrence=cooccurrence,
                                z_k=z_k,
                                opt=opt,
                                batch_gibbs=True,
                                scale=scale)
    model.train(outputpath,
                epochs=epochs,
                batches=batches,
                steps=steps,
                gibbs_batch_size=gibbs_batch_size)


if __name__ == "__main__":
    main()
