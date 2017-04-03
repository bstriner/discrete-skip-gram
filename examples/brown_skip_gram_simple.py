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
    skip_gram_ones_generator
from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.callbacks.write_encodings import WriteEncodings
from discrete_skip_gram.models.word_dqn import WordDQN


def main():
    outputpath="output/brown_skip_gram_simple"
    min_count = 20
    z_depth = 5
    z_k = 8
    units = 256
    batch_size = 64
    window = 5
    samples = 1
    batches = 256
    decoder_batches = 4
    value_batches = 4
    epochs = 1000

    docs = brown_docs()
    docs = clean_docs(docs, simple_clean)
    count = sum(len(doc) for doc in docs)
    wordcounts = count_words(docs)
    wordset = [k for k, v in wordcounts.iteritems() if v >= min_count]
    wordset.sort()
    wordmap = {w: i for i, w in enumerate(wordset)}
    adocs = docs_to_arrays(docs, wordmap)

    k = len(wordset) + 1

    print "Total wordcount: {}".format(count)
    print "Unique words: {}, Filtered: {}".format(len(wordcounts), len(wordset))

    # hidden = 64
    x = Input((1,), dtype='int32')
    ex = Embedding(k, z_k)
    rsx = Reshape((z_k,))
    # dx = Dense(z_k)
    sm = Activation('softmax')
    z = sm(rsx(ex(x)))  # n, z_k

    ones = Lambda(lambda _x: T.ones((_x.shape[0], 1)), output_shape=lambda _x: (_x[0], 1))
    dy = Dense(z_k * k)
    rsy = Reshape((z_k, k))
    smy = softmax_nd_layer()
    y = smy(rsy(dy(ones(x))))  # n, z_k, k

    yreal = Input((1,), dtype='int32')
    # def one_hot(_yreal):
    #    ret = T.zeros((_yreal.shape[0], k), dtype=np.float32)
    #    ret = T.set_subtensor(ret[T.arange(ret.shape[0]), T.flatten(_yreal)], 1)
    #    return ret
    # yreal_onehot = Lambda(one_hot, output_shape=lambda _yreal: (_yreal[0], 1))(yreal)  # n, k

    lossl = Lambda(lambda (_y, _yreal):
                   -T.log(_y[T.arange(_y.shape[0]), :, T.flatten(_yreal)]),
                   #                   T.sum(T.sum(-_yreal.dimshuffle((0, 'x', 1)) * T.log(_out), axis=1), axis=1, keepdims=True) +
                   #                   T.sum(T.sum(-(1 - _yreal.dimshuffle((0, 'x', 1))) * T.log(1 - _out), axis=1), axis=1, keepdims=True),
                   output_shape=lambda (_out, _yreal): (_out[0], _out[1]))
    loss = lossl([y, yreal])  # n, z_k

    weighted_loss = Lambda(lambda (_loss, _z): T.sum(_loss * _z, axis=1, keepdims=True),
                           output_shape=lambda _x: (_x[0], 1))([loss, z])  # n, 1

    initial_reg_weight = 0
    if initial_reg_weight > 0:
        reg_weight = K.variable(initial_reg_weight, dtype='float32', name='reg_weight')
        #reg = reg_weight * T.mean(T.sum(T.log(z)+T.log(1-z), axis=1), axis=0)
        reg = -reg_weight * T.mean(T.sum(T.square(z-0.5), axis=1), axis=0)
        sm.add_loss(reg)

    model_nll = Model(inputs=[x, yreal], output=[weighted_loss])
    model_nll.compile(Adam(1e-3), lambda ytrue, ypred: ypred)

    model_zp = Model(inputs=[x], output=[z])

    model_nll.summary()
    model_zp.summary()

    gen = skip_gram_ones_generator(adocs, window, batch_size)

    def on_epoch_end(epoch, logs):
        p = "{}/epoch-{:08d}.txt".format(outputpath,epoch)
        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))
        n = 64
        x, _ = skip_gram_batch(adocs, window, n)
        zp = model_zp.predict(x)
        with open(p, 'w') as f:
            for i in range(n):
                w = "_UNK_" if x[i, 0] == 0 else wordset[x[i, 0] - 1]
                f.write("{}: {}\n".format(w, ", ".join("{:.03f}".format(_z) for _z in zp[i, :])))
        if (epoch + 1) % 10 == 0:
            p = "{}/vocab-{:08d}.csv".format(outputpath,epoch)
            x = np.arange(k, dtype=np.int32).reshape((-1, 1))
            zp = model_zp.predict(x, verbose=0)
            am = np.argmax(zp, axis=1)
            maxima = np.max(zp, axis=1)
            with open(p, 'wb') as f:
                w = csv.writer(f)
                headings = ["Group {}".format(i) for i in range(z_k)]
                w.writerow(['word', 'argmax', 'max'] + headings)
                for i in range(k):
                    word = "_unk_" if i == 0 else wordset[i - 1]
                    l = [zp[i, j] for j in range(z_k)]
                    w.writerow([word, am[i], maxima[i]] + l)

    cb = LambdaCallback(on_epoch_end=on_epoch_end)
    csvcb = CSVLogger("{}/history.csv".format(outputpath))
    model_nll.fit_generator(gen, epochs=500, steps_per_epoch=1024, callbacks=[cb, csvcb])


if __name__ == "__main__":
    main()
