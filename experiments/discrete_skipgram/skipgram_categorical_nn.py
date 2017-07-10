import os
#os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

from discrete_skip_gram.dataset_util import load_dataset
from discrete_skip_gram.skipgram.categorical_nn import CategoricalNNModel
from discrete_skip_gram.skipgram.corpus import load_corpus
from discrete_skip_gram.skipgram.dataset import load_dataset
from keras.optimizers import Adam
from keras.regularizers import L1L2


def main():
    batch_size = 99999999
    opt = Adam(3e-4)
    regularizer = L1L2(1e-12, 1e-12)
    outputpath = "output/skipgram_categorical_co"
    z_k = 1024
    epochs = 1000
    batches = 64
    dataset_path = "output/dataset.npy"
    dataset = load_dataset(dataset_path)
    corpus_path = "output/corpus"
    vocab, corpus = load_corpus(corpus_path)

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    model = CategoricalNNModel(x_k=len(vocab)+1, z_k=z_k, opt=opt, regularizer=regularizer)
    model.train(outputpath=outputpath, epochs=epochs, batches=batches, batch_size=batch_size, dataset=dataset)


if __name__ == "__main__":
    main()
