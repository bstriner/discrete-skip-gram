import numpy as np

from discrete_skip_gram.skipgram.categorical_col import train_model
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.regularizers import BalanceRegularizer
from keras.optimizers import Adam


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    opt = Adam(1e-3)
    epochs = 50
    batches = 1024
    z_k = 1024
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    for name, weight in [("1e2", 1e-2),
                         ("1e3", 1e-3),
                         ("1e4", 1e-4),
                         ("1e5", 1e-5),
                         ("1e6", 1e-6),
                         ("1e7", 1e-7)]:
        train_model(
            outputpath="output/skipgram_flat-b-{}".format(name),
            pz_regularizer=BalanceRegularizer(weight),
            cooccurrence=cooccurrence,
            z_k=z_k,
            opt=opt,
            epochs=epochs,
            batches=batches)


if __name__ == "__main__":
    main()
