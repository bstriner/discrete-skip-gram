import numpy as np

from discrete_skip_gram.skipgram.categorical_col import train_model
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.regularizers import ExclusiveLasso
from keras.optimizers import Adam


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    opt = Adam(1e-3)
    epochs = 50
    batches = 1024
    z_k = 1024
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    for name, weight in [("1e7", 1e-7), ("1e8", 1e-8), ("1e9", 1e-9), ("1e10", 1e-10), ("1e11", 1e-11)]:
        train_model(
            outputpath="output/skipgram_flat-el-{}".format(name),
            pz_weight_regularizer=ExclusiveLasso(weight),
            cooccurrence=cooccurrence,
            z_k=z_k,
            opt=opt,
            epochs=epochs,
            batches=batches)


if __name__ == "__main__":
    main()
