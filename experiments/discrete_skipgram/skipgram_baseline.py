import os

from discrete_skip_gram.skipgram.baseline import BaselineModel
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from keras.optimizers import Adam
from keras.regularizers import L1L2


# os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"


def main():
    outputpath = "output/skipgram_baseline"
    opt = Adam(1e-3)
    z_units = 256
    regularizer = L1L2(1e-8, 1e-8)
    epochs = 1000
    batches = 5000
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy')
    model = BaselineModel(
        cooccurrence=cooccurrence,
        z_units=z_units,
        opt=opt,
        regularizer=regularizer,
    )
    model.train(outputpath=outputpath,
                epochs=epochs,
                batches=batches)


if __name__ == "__main__":
    main()
