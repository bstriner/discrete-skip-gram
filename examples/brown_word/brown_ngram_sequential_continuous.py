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
from discrete_skip_gram.datasets.utils import clean_docs, simple_clean, get_words, count_words, \
    format_encoding_sequential_continuous
from discrete_skip_gram.datasets.word_dataset import docs_to_arrays, skip_gram_generator, skip_gram_batch, \
    skip_gram_ones_generator, WordDataset
from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.models.word_ngram_sequential_continuous import WordNgramSequentialContinuous
from discrete_skip_gram.models.util import makepath
from keras.regularizers import L1L2


def main():
    outputpath = "output/brown/ngram_sequential_continuous"
    min_count = 5
    batch_size = 128
    epochs = 1000
    steps_per_epoch = 256
    window = 3
    hidden_dim = 256
    z_k = 2
    z_depth = 6
    #4^6 = 4096
    decay = 0.9
    reg = L1L2(1e-6, 1e-6)
    lr = 3e-4

    docs = clean_docs(brown_docs(), simple_clean)
    docs, tdocs = docs[:-5], docs[-5:]
    ds = WordDataset(docs, min_count, tdocs=tdocs)
    ds.summary()
    k = ds.k
    schedule = np.power(decay, np.arange(z_depth))
    model = WordNgramSequentialContinuous(ds, z_k=z_k, z_depth=z_depth, schedule=schedule,
                                          reg=reg,
                                          hidden_dim=hidden_dim, window=window, lr=lr)
    csvpath = "{}/history.csv".format(outputpath)
    makepath(csvpath)
    csvcb = CSVLogger(csvpath)
    validation_n = 4096
    vd = ds.cbow_batch(n=validation_n, window=window, test=True)

    def on_epoch_end(epoch, logs):
        path = "{}/generated-{:08d}.txt".format(outputpath, epoch)
        n = 128
        _, x = ds.cbow_batch(n=n, window=window, test=True)
        y = model.model_predict.predict(x, verbose=0)
        with open(path, 'w') as f:
            for i in range(n):
                w = ds.get_word(x[i, 0])
                ctx = [ds.get_word(y[i, j]) for j in range(window * 2)]
                lctx = " ".join(ctx[:window])
                rctx = " ".join(ctx[window:])
                f.write("{} [{}] {}\n".format(lctx, w, rctx))

        if (epoch + 1) % 10 == 0:
            path = "{}/encoded-{:08d}.csv".format(outputpath, epoch)
            x = np.arange(k).reshape((-1, 1))
            z = model.model_encode.predict(x, verbose=0)
            znorm = z - np.mean(z, axis=0, keepdims=True)
            with open(path, 'w') as f:
                w = csv.writer(f)
                w.writerow(["Idx", "Word", "Encoding"])
                for i in range(k):
                    word = ds.get_word(i)
                    enc = format_encoding_sequential_continuous(znorm[i, :, :])
                    w.writerow([i, word, enc])
            path = "{}/encoded-array-{:08d}.txt".format(outputpath, epoch)
            np.save(path, z)

    gencb = LambdaCallback(on_epoch_end=on_epoch_end)
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=([vd[1], vd[0]], np.ones((validation_n, 1), dtype=np.float32)),
                callbacks=[csvcb, gencb])


if __name__ == "__main__":
    main()
