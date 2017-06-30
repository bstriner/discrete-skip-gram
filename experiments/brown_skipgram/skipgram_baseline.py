# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

from discrete_skip_gram.dataset_util import load_dataset
from discrete_skip_gram.layers.utils import leaky_relu
from discrete_skip_gram.skipgram_models.skipgram_baseline_model import SkipgramBaselineModel
from keras.regularizers import L1L2


def main():
    outputpath = "output/skipgram_baseline"
    ds = load_dataset()
    batch_size = 256
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 50
    embedding_units = 512
    kernel_regularizer = L1L2(1e-9, 1e-9)
    embeddings_regularizer = L1L2(1e-9, 1e-9)
    lr = 3e-4
    model = SkipgramBaselineModel(dataset=ds,
                                  window=window,
                                  embedding_units=embedding_units,
                                  kernel_regularizer=kernel_regularizer,
                                  embeddings_regularizer=embeddings_regularizer,
                                  inner_activation=leaky_relu,
                                  lr=lr)
    model.summary()

    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                output_path=outputpath,
                frequency=frequency)


if __name__ == "__main__":
    main()
