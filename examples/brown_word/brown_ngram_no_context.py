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
from discrete_skip_gram.models.word_ngram_no_context import WordNgramNoContext
from discrete_skip_gram.models.util import makepath


def main():
    outputpath = "output/brown/ngram_no_context"
    min_count = 5
    batch_size = 128
    epochs = 100
    steps_per_epoch = 1024
    window = 3
    hidden_dim = 256

    docs = clean_docs(brown_docs(), simple_clean)
    docs, tdocs = docs[:-5], docs[-5:]
    ds = WordDataset(docs, min_count, tdocs=tdocs)
    ds.summary()
    model = WordNgramNoContext(ds, hidden_dim=hidden_dim, window=window, lr=1e-3)
    csvpath = "{}/history.csv".format(outputpath)
    makepath(csvpath)
    csvcb = CSVLogger(csvpath)
    validation_n = 4096
    vd = ds.cbow_batch(n=validation_n, window=window, test=True)
    def on_epoch_end(epoch, logs):
        path = "{}/generated-{:08d}.txt".format(outputpath, epoch)
        n = 128
        x = np.zeros((n,1), dtype=np.float32)
        y = model.model_predict.predict(x, verbose=0)
        with open(path, 'w') as f:
            for i in range(n):
                ctx = [ds.get_word(y[i, j]) for j in range(window * 2)]
                lctx = " ".join(ctx[:window])
                rctx = " ".join(ctx[window:])
                f.write("{} [SKIP] {}\n".format(lctx, rctx))
    gencb = LambdaCallback(on_epoch_end=on_epoch_end)
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=(vd[0], np.ones((validation_n, 1), dtype=np.float32)),
                callbacks=[csvcb, gencb])


if __name__ == "__main__":
    main()
