import numpy as np
from keras.optimizers import Adam

from discrete_skip_gram.flat_nn_model import FlatNNModel
from discrete_skip_gram.initializers import uniform_initializer


def main():
    epochs = 10
    batches = 4096
    z_k = 1024
    initializer = uniform_initializer(0.05)
    outputpath = "output/skipgram_1024_nn"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    initial_b = np.log(np.sum(cooccurrence, axis=0))
    opt = Adam(1e-3)
    model = FlatNNModel(
        cooccurrence=cooccurrence,
        z_k=z_k,
        opt=opt,
        initializer=initializer,
        initial_b=initial_b
    )
    model.train(outputpath=outputpath,
                epochs=epochs,
                batches=batches)


if __name__ == "__main__":
    main()
