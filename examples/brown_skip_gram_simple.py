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


def main():
    outputpath = "output/brown_skip_gram_simple"
    min_count = 5
    z_k = 8
    batch_size = 128
    window = 3
    epochs = 500
    steps_per_epoch = 32
    docs = clean_docs(brown_docs(), simple_clean)
    ds = WordDataset(docs, min_count)
    ds.summary()
    k = ds.k

    # hidden = 64
    x = Input((1,), dtype='int32')
    ex = Embedding(k, z_k)
    rsx = Reshape((z_k,))
    # dx = Dense(z_k)
    sm = Activation('softmax')
    z = sm(rsx(ex(x)))  # n, z_k

    ones = Lambda(lambda _x: T.ones((_x.shape[0], 1)), output_shape=lambda _x: (_x[0], 1))
    dy = Dense(z_k * ds.k)
    rsy = Reshape((z_k, ds.k))
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

    eps = K.epsilon()
    initial_reg_weight = 1e-3
    if initial_reg_weight > 0:
        reg_weight = K.variable(initial_reg_weight, dtype='float32', name='reg_weight')
        reg = reg_weight * T.mean(T.sum(T.log(z + eps) + T.log(1 - z + eps), axis=1), axis=0)
        # reg = -reg_weight * T.mean(T.sum(T.square(z-0.5), axis=1), axis=0)
        sm.add_loss(reg)

    balance_reg = 1e-2
    if balance_reg > 0:
        balance_reg = K.variable(np.float32(balance_reg), dtype='float32', name='balance_reg')
        reg = T.mean(-T.log(T.mean(z, axis=0)+eps), axis=0) * balance_reg
        sm.add_loss(reg)

    def certainty(y_true, y_pred):
        return T.mean(T.max(z, axis=1), axis=0)

    def nll(y_true, y_pred):
        return T.mean(weighted_loss, axis=None)

    def kl(y_true, y_pred):
        return T.mean(-T.log(T.mean(z, axis=0)), axis=0)

    model_nll = Model(inputs=[x, yreal], output=[weighted_loss])
    model_nll.compile(Adam(1e-3), lambda ytrue, ypred: ypred, metrics=[certainty, nll, kl])

    model_zp = Model(inputs=[x], output=[z])

    model_nll.summary()
    model_zp.summary()

    gen = skip_gram_ones_generator(ds.adocs, window, batch_size)

    def on_epoch_end(epoch, logs):
        p = "{}/epoch-{:08d}.txt".format(outputpath, epoch)
        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))
        n = 64
        x, _ = skip_gram_batch(ds.adocs, window, n)
        zp = model_zp.predict(x)
        with open(p, 'w') as f:
            for i in range(n):
                w = ds.get_word(x[i, 0])
                f.write("{}: {}\n".format(w, ", ".join("{:.03f}".format(_z) for _z in zp[i, :])))
        if (epoch + 1) % 5 == 0:
            p = "{}/vocab-{:08d}.csv".format(outputpath, epoch)
            x = np.arange(ds.k, dtype=np.int32).reshape((-1, 1))
            zp = model_zp.predict(x, verbose=0)
            am = np.argmax(zp, axis=1)
            maxima = np.max(zp, axis=1)
            with open(p, 'wb') as f:
                w = csv.writer(f)
                headings = ["Group {}".format(i) for i in range(z_k)]
                w.writerow(['word', 'argmax', 'max'] + headings)
                for i in range(ds.k):
                    word = ds.get_word(i)
                    l = [zp[i, j] for j in range(z_k)]
                    w.writerow([word, am[i], maxima[i]] + l)

    cb = LambdaCallback(on_epoch_end=on_epoch_end)
    csvp = "{}/history.csv".format(outputpath)
    if not os.path.exists(os.path.dirname(csvp)):
        os.makedirs(os.path.dirname(csvp))
    csvcb = CSVLogger(csvp, append=True)
    model_nll.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[cb, csvcb])


if __name__ == "__main__":
    main()
