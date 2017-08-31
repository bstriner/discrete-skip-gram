import numpy as np
from discrete_skip_gram.flat_train import train_flat_model
from discrete_skip_gram.regularizers import BalanceRegularizer
from keras.optimizers import Adam, RMSprop
from discrete_skip_gram.watchdog import Watchdog
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    epochs = 1000
    batches = 4096
    z_k = 1024
    outputpath = "output/skipgram_flat_tst"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    #reg = BalanceRegularizer(1e-4)
    scale = 1e-2
    reg = None
    watchdog = Watchdog(limit=1e-6, iters=500, path="{}/watchdog.txt".format(outputpath))
    train_flat_model(outputpath=outputpath,
                     epochs=epochs,
                     batches=batches,
                     cooccurrence=cooccurrence,
                     z_k=z_k,
                     watchdog=watchdog,
                     opt=Adam(1e-3),
                     scale=scale,
                     reset_n=100,
                     pz_regularizer=reg)


if __name__ == "__main__":
    main()
