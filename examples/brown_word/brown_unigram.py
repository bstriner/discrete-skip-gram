from keras.layers import Input, Embedding, Dense, Reshape, Activation, Lambda
from keras.models import Model
from keras.callbacks import LambdaCallback, CSVLogger
from keras import backend as K
import os
# os.environ["THEANO_FLAGS"]="optimizer=None"
import csv
import theano.tensor as T

from nltk.corpus import brown
import itertools
from nltk.stem.porter import PorterStemmer
import numpy as np
from keras.optimizers import Adam
from discrete_skip_gram.layers.utils import softmax_nd_layer
from discrete_skip_gram.datasets.utils import clean_docs, simple_clean, get_words, count_words, format_encoding
from discrete_skip_gram.datasets.word_dataset import docs_to_arrays, skip_gram_generator, skip_gram_batch, \
    skip_gram_ones_generator, WordDataset
from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.callbacks.write_encodings import WriteEncodings
from discrete_skip_gram.models.word_dqn import WordDQN
from discrete_skip_gram.models.word_unigram import WordUnigram
from discrete_skip_gram.models.util import makepath


def main():
    outputpath = "output/brown/unigram"
    min_count = 5
    batch_size = 64
    epochs = 1000
    steps_per_epoch = 1024

    docs = clean_docs(brown_docs(), simple_clean)
    ds = WordDataset(docs, min_count)
    ds.summary()
    model = WordUnigram(ds)
    csvpath = "{}/history.csv".format(outputpath)
    makepath(csvpath)
    cb = CSVLogger(csvpath)
    model.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[cb])


if __name__ == "__main__":
    main()
