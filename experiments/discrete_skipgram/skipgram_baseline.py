import os

from discrete_skip_gram.skipgram.baseline import BaselineModel
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from keras.optimizers import Adam
from keras.regularizers import L1L2


# os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"


def main():
    op = "output/skipgram_baseline"
    epochs = 20
    batches = 5000
    cooccurrence = load_cooccurrence('output/cooccurrence.npy')
    for z_units in [512, 256, 128, 64, 32]:
        outputpath="{}/{}".format(op, z_units)
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        model = BaselineModel(
            cooccurrence=cooccurrence,
            z_units=z_units,
            opt=Adam(1e-3),
            regularizer=L1L2(1e-10, 1e-10),
        )
        model.train(outputpath=outputpath,
                epochs=epochs,
                batches=batches)


if __name__ == "__main__":
    main()
