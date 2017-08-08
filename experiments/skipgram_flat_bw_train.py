import numpy as np
from discrete_skip_gram.flat_train import train_flat_regularizer_battery
from discrete_skip_gram.regularizers import BalanceWeightedRegularizer


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    epochs = 10
    batches = 4096
    z_k = 1024
    iters = 1
    outputpath = "output/skipgram_flat-bw"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    weights = [
        1e-1,
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
        1e-7
    ]
    marginal = np.sum(cooccurrence, axis=1) / np.sum(cooccurrence, axis=None)
    regularizers = [BalanceWeightedRegularizer(w, marginal) for w in weights]
    labels = ["bw-{:.01e}".format(w) for w in weights]
    train_flat_regularizer_battery(
        outputpath=outputpath,
        cooccurrence=cooccurrence,
        epochs=epochs,
        batches=batches,
        iters=iters,
        z_k=z_k,
        labels=labels,
        regularizers=regularizers,
        is_weight_regularizer=True,
        kwdata={weights: np.array(weights)}
    )


if __name__ == "__main__":
    main()
