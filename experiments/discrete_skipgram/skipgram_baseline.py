# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

from discrete_skip_gram.dataset_util import load_dataset
from discrete_skip_gram.layers.utils import leaky_relu
from discrete_skip_gram.skipgram.corpus import load_corpus
from discrete_skip_gram.skipgram.dataset import load_dataset
from discrete_skip_gram.skipgram_models.skipgram_baseline_model import SkipgramBaselineModel
from keras.optimizers import Adam
from keras.regularizers import L1L2


def main():
    outputpath = "output/skipgram_baseline"
    dataset_path = "output/dataset.npy"
    corpus_path = "output/corpus"
    vocab, corpus = load_corpus(corpus_path)
    dataset = load_dataset(dataset_path)

    batch_size = 512
    epochs = 5000
    frequency = 5
    embedding_units = 1024
    l1 = 1e-11
    l2 = 1e-11
    kernel_regularizer = L1L2(l1, l2)
    embeddings_regularizer = L1L2(l1, l2)
    opt = Adam(3e-5)
    x_k = len(vocab) + 1
    model = SkipgramBaselineModel(x_k=x_k,
                                  embedding_units=embedding_units,
                                  kernel_regularizer=kernel_regularizer,
                                  embeddings_regularizer=embeddings_regularizer,
                                  inner_activation=leaky_relu,
                                  opt=opt)
    model.summary()

    model.train(x=dataset, batch_size=batch_size,
                epochs=epochs,
                output_path=outputpath,
                frequency=frequency)


if __name__ == "__main__":
    main()
