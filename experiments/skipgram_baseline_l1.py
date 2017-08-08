from discrete_skip_gram.skipgram.baseline import run_baseline
from keras.regularizers import l1

from discrete_skip_gram.cooccurrence import load_cooccurrence


# os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"


def main():
    op = "output/skipgram_baseline-l1"
    epochs = 50
    batches = 4096
    cooccurrence = load_cooccurrence('output/cooccurrence.npy')
    z_ks = [512, 256, 128, 64, 32]
    iters = 5
    run_baseline(cooccurrence=cooccurrence,
                 z_ks=z_ks,
                 iters=iters,
                 output_path=op,
                 epochs=epochs,
                 batches=batches,
                 regularizer=l1(1e-8))


if __name__ == "__main__":
    main()
