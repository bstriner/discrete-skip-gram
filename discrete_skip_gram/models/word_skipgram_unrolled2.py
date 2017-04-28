"""
Each element of sequence is an embedding layer
"""

import csv

import numpy as np
import theano
from keras.callbacks import LambdaCallback, CSVLogger
from keras.layers import Input, Embedding, Lambda, Activation, Add
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..layers.unrolled.bias_layer import BiasLayer
from ..layers.unrolled.decoder_layer import DecoderLayer
from ..layers.unrolled.sampler_deterministic_layer import SamplerDeterministicLayer
from ..layers.unrolled.skipgram_batch_layer import SkipgramBatchLayer, SkipgramBatchPolicyLayer
from ..layers.utils import drop_dim_2, zeros_layer, add_layer


def selection_layer(zind):
    return Lambda(lambda (_a, _b): _a * (_b[:, zind].dimshuffle((0, 'x'))), output_shape=lambda (_a, _b): _a)


class WordSkipgramUnrolled2(object):
    def __init__(self, dataset, units, window, z_depth, z_k,
                 lr=1e-4,
                 act_reg=None,
                 reg=None,
                 balance_reg=0,
                 certainty_reg=0
                 ):
        self.dataset = dataset
        self.units = units
        self.window = window
        self.z_depth = z_depth
        self.z_k = z_k
        self.y_depth = window * 2
        assert z_depth > 0
        srng = RandomStreams(123)
        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((self.y_depth,), dtype='int32', name='input_y')

        # ys = [Lambda(lambda _y: _y[:, i:i + 1], output_shape=lambda _y: (_y[0], 1))(input_y) for i in
        #      range(self.y_depth)]

        pzs = []
        zhs = []
        zs = []
        losses = []
        decoder_h0 = BiasLayer(units)
        ht = decoder_h0(input_x)
        zt = zeros_layer(1, dtype='int32')(input_x)
        eps = 1e-6
        sampler = SamplerDeterministicLayer(offset=0)
        losses = []

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        if balance_reg > 0:
            self.balance_reg = theano.shared(np.float32(balance_reg), name='balance_reg')
        if certainty_reg > 0:
            self.certainty_reg = theano.shared(np.float32(certainty_reg), name='certainty_reg')
        decoder_layer = DecoderLayer(units, z_k=z_k + 1, kernel_regularizer=reg,
                                     name="decoder")
        skipgram_layer = SkipgramBatchLayer(units=units, y_k=x_k, z_k=z_k, kernel_regularizer=reg,
                                            name="skipgram")
        for idx in range(z_depth):

            # predictions
            ht, zh = decoder_layer([ht, zt])
            zhs.append(zh)
            nll = skipgram_layer([zh, input_y])

            # embed x
            embedding = Embedding(x_k, z_k, name="embedding_x_{}".format(idx))
            pz = Activation("softmax")(drop_dim_2()(embedding(input_x)))
            pzs.append(pz)
            z = sampler(pz)
            zs.append(z)
            zt = add_layer(1)(z)

            # loss calc
            loss = Lambda(lambda (_a, _b): T.sum(_a * _b, axis=1, keepdims=True),
                          output_shape=lambda (_a, _b): (_a[0], 1))([nll, pz])
            losses.append(loss)

            if balance_reg > 0:
                decoder_layer.add_loss(self.balance_reg * T.mean(-T.log(T.mean(pz, axis=0) + eps), axis=None))
            if certainty_reg > 0:
                decoder_layer.add_loss(self.certainty_reg * T.mean(T.log(1 - pz + eps) + T.log(pz + eps), axis=None))

        opt = Adam(lr)
        if self.z_depth == 1:
            loss = losses[0]
        else:
            loss = Add()(losses)
        self.model = Model([input_x, input_y], loss)

        def nll_initial(_yt, _yp):
            return T.mean(losses[0], axis=None)

        def nll_final(_yt, _yp):
            return T.mean(losses[-1], axis=None)

        self.model.compile(opt, loss_f, metrics=[nll_initial, nll_final])
        self.model._make_train_function()

        # Prediction model
        policy_layer = SkipgramBatchPolicyLayer(skipgram_layer, srng=srng, depth=self.y_depth)
        ygen = policy_layer([zh, z])
        self.predict_model = Model([input_x], ygen)

        # Encoder model
        self.encode_model = Model([input_x], pzs + zs)

        # Decoder model
        input_zs = [Input((1,), dtype='int32', name="input_z_{}".format(i)) for i in range(z_depth)]
        ht = decoder_h0(input_zs[0])
        zt = zeros_layer(1, dtype='int32')(input_zs[0])
        for z in input_zs:
            ht, zhdec = decoder_layer([ht, zt])
            zt = add_layer(1)(z)
        ygen = policy_layer([zhdec, z])
        self.decoder_model = Model(input_zs, ygen)

    def train(self, batch_size, epochs, steps_per_epoch, output_path, **kwargs):
        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)

        def on_epoch_end(epoch, logs):
            if (epoch + 1) % 10 == 0:
                path = "{}/generated-{:08d}.txt".format(output_path, epoch)
                n = 128
                samples = 8
                _, x = self.dataset.cbow_batch(n=n, window=self.window, test=True)
                ys = [self.predict_model.predict(x, verbose=0) for _ in range(samples)]
                with open(path, 'w') as f:
                    for i in range(n):
                        strs = []
                        w = self.dataset.get_word(x[i, 0])
                        for y in ys:
                            ctx = [self.dataset.get_word(y[i, j]) for j in range(self.window * 2)]
                            lctx = " ".join(ctx[:self.window])
                            rctx = " ".join(ctx[self.window:])
                            strs.append("{} [{}] {}".format(lctx, w, rctx))
                        f.write("{}: {}\n".format(w, " | ".join(strs)))

                path = "{}/encoded-{:08d}.csv".format(output_path, epoch)
                x = np.arange(self.dataset.k).reshape((-1, 1))
                ret = self.encode_model.predict(x, verbose=0)
                pzs, zs = ret[:self.z_depth], ret[self.z_depth:]

                # if z_depth == 1:
                #    zs = [zs]
                with open(path, 'wb') as f:
                    w = csv.writer(f)
                    w.writerow(["Idx", "Word", "Encoding"] +
                               ["Cat {}".format(i) for i in range(len(zs))] +
                               ["Pz {}".format(i) for i in range(len(zs))])
                    for i in range(self.dataset.k):
                        word = self.dataset.get_word(i)
                        enc = [z[i, 0] for z in zs]
                        pzfs = [", ".join("{:03f}".format(p) for p in pz[i, :]) for pz in pzs]
                        encf = "".join(chr(ord('a') + e) for e in enc)
                        w.writerow([i, word, encf] + enc + pzfs)
                path = "{}/weights-{:08d}.h5".format(output_path, epoch)
                self.model.save_weights(path)

        csvpath = "{}/history.csv".format(output_path)
        cbs = [LambdaCallback(on_epoch_begin=on_epoch_end), CSVLogger(csvpath)]
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=cbs,
                                 verbose=1, **kwargs)
