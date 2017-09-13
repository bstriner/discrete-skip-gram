import numpy as np
from discrete_skip_gram.flat_train import train_flat_battery


def main():
    epochs = 10
    iters = 5
    batches = 4096
    z_k = 256
    outputpath = "output/skipgram_256"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    train_flat_battery(
        outputpath=outputpath,
        epochs=epochs,
        batches=batches,
        z_k=z_k,
        iters=iters,
        cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
