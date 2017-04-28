import csv
import os
import numpy as np
import theano
from keras.layers import Input, Embedding, Lambda, Add
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from keras.callbacks import LambdaCallback, CSVLogger
from theano.tensor.shared_randomstreams import RandomStreams

from ..layers.unrolled.bias_layer import BiasLayer
from ..layers.unrolled.decoder_layer import DecoderLayer
from ..layers.unrolled.encoder_layer import EncoderLayer
from ..layers.unrolled.sampler_layer import SamplerLayer
from ..layers.unrolled.skipgram_batch_layer import SkipgramBatchLayer, SkipgramBatchPolicyLayer
from ..layers.unrolled.sampler_deterministic_layer import SamplerDeterministicLayer
from ..layers.utils import drop_dim_2, zeros_layer, ones_layer, add_layer

"""
Each word is embedded as a series of independent steps. p(xz0|x)...p(xzn|x).
Decoder outputs p(yz|xz).
Adversary learns p(zt|z0...zt-1).
Objective is maximize -log(p(y|yz)p(yz|xz)p(xz|x))
p(y|yz) = p(yz|y)*p(y)/p(yz)
"""

def selection_layer(zind):
    return Lambda(lambda (_a, _b): _a * (_b[:, zind].dimshuffle((0, 'x'))), output_shape=lambda (_a, _b): _a)

class WordSkipgramUnrolledBatch(object):
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

        # embed x
        embedding = Embedding(x_k, units)
        embedded_x = drop_dim_2()(embedding(input_x))

        # encode to z
        encoder_h0 = BiasLayer(units)
        pzs = []
        zs = []
        ht = encoder_h0(input_x)
        zt = zeros_layer(1, dtype='int32')(input_x)
        encoder_layer = EncoderLayer(units=units, z_k=z_k, kernel_regularizer=reg)
        sampler = SamplerLayer(srng, offset=1)
        for i in range(z_depth):
            # print "Depth: {}".format(i)
            ht, pz = encoder_layer([ht, zt, embedded_x])
            zt = sampler(pz)
            pzs.append(pz)
            """
            _n = 16
            _x = np.random.randint(low=0, high=x_k, size=(_n, 1))
            _y = np.random.randint(low=0, high=x_k, size=(_n, self.y_depth))
            _m = Model([input_x],[pz])
            _pz = _m.predict([_x])
            print "_pz Shape: {} / {}".format(_pz.shape, pz._keras_shape)
            print "PZ: {}".format(_pz)
            raise ValueError("{}".format(pz))
            """
            # print "PZ shape: {}".format(pz._keras_shape)
            zs.append(add_layer(-1)(zt))

        # decode to zh
        zhs = []
        losses = []
        decoder_h0 = BiasLayer(units)
        ht = decoder_h0(input_x)
        zt = zeros_layer(1, dtype='int32')(input_x)
        decoder_layer = DecoderLayer(units, z_k=z_k+1, kernel_regularizer=reg)
        skipgram_layer = SkipgramBatchLayer(units=units, y_k=x_k, z_k=z_k, kernel_regularizer=reg)
        for zidx, z in enumerate(zs):
            ht, zh = decoder_layer([ht, zt])
            nll = skipgram_layer([zh, input_y]) #n, z_k
            loss = Lambda(lambda (_a, _b): T.sum(_a * _b, axis=1, keepdims=True),
                          output_shape=lambda (_a, _b):(_a[0],1))([nll, pzs[zidx]])
            losses.append(loss)
            zt = add_layer(1)(z)
            zhs.append(zh)

        # nllconcat = Concatenate()(losses)
        # print "NLLCONCAT: {}".format(nllconcat._keras_shape)
        # nlltot = Lambda(lambda _nll: T.sum(_nll * (self.schedule.dimshuffle(('x', 0))), axis=1, keepdims=True),
        #                output_shape=lambda _nll: (_nll[0], 1))(nllconcat)
        if z_depth > 1:
            nlltot = Add()(losses)
        else:
            nlltot = losses[0]

        def nll_initial(ytrue, ypred):
            return T.mean(losses[0], axis=None)

        def nll_final(ytrue, ypred):
            return T.mean(losses[-1], axis=None)

        eps = 1e-6
        if balance_reg > 0:
            self.balance_reg = theano.shared(np.float32(balance_reg), name='balance_reg')
            for pz in pzs:
                # p = (1.0/self.z_k)
                skipgram_layer.add_loss(self.balance_reg * T.mean(-T.log(T.mean(pz, axis=0) + eps), axis=None))
        if certainty_reg > 0:
            self.certainty_reg = theano.shared(np.float32(certainty_reg), name='certainty_reg')
            for pz in pzs:
                skipgram_layer.add_loss(self.certainty_reg * T.mean(T.log(1 - pz + eps) + T.log(pz + eps), axis=None))

        opt = Adam(lr)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        self.model = Model([input_x, input_y], nlltot)
        self.model.compile(opt, loss_f, metrics=[nll_initial, nll_final])

        """
        _n = 128
        _x = np.random.randint(low=0, high=x_k, size=(_n, 1))
        _y = np.random.randint(low=0, high=x_k, size=(_n, self.y_depth))
        _nll = self.model.predict([_x, _y])
        print "NLL Shape: {} / {}".format(_nll.shape, nlltot._keras_shape)
        """
        # Prediction model
        policy_layer = SkipgramBatchPolicyLayer(skipgram_layer, srng=srng, depth=self.y_depth)
        ygen = policy_layer([zhs[-1], zs[-1]])
        self.predict_model = Model([input_x], ygen)

        # Encoder model
        sampler_det = SamplerDeterministicLayer(offset=1)
        zdets = []
        pzdets = []
        ht = encoder_h0(input_x)
        zt = zeros_layer(1, dtype='int32')(input_x)
        for i in range(z_depth):
            ht, pz = encoder_layer([ht, zt, embedded_x])
            zt = sampler_det(pz)
            pzdets.append(pz)
            zdets.append(add_layer(-1)(zt))
        self.encode_model = Model([input_x], pzdets + zdets)

        # Decoder model
        input_zs = [Input((1,), dtype='int32', name="input_z_{}".format(i)) for i in range(z_depth)]

        ht = decoder_h0(input_zs[0])
        zt = zeros_layer(1, dtype='int32')(input_zs[0])
        #        decoder_layer = DecoderLayer(units, z_k=z_k, kernel_regularizer=reg)
        #        skipgram_layer = SkipgramLayer(units=units, k=x_k, kernel_regularizer=reg)
        #        policy_layer = SkipgramPolicyLayer(skipgram_layer, srng=srng, depth=self.y_depth)

        for z in input_zs:
            ht, zhdec = decoder_layer([ht, zt])
            zt = add_layer(1)(z)

        ygen = policy_layer([zhdec, z])
        self.decoder_model = Model(input_zs, ygen)

    def write_encodings(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word", "Encoding"] + ["P{}".format(j) for j in range(self.z_depth)])
            x = np.arange(self.dataset.k).reshape((-1, 1))
            ret = self.encode_model.predict(x, verbose=0)
            print "Shapes: {}".format(r.shape for r in ret)

            pzs, zs = ret[:self.z_depth], ret[self.z_depth:]
            for i in range(self.dataset.k):
                enc = [z[i, 0] for z in zs]
                ps = [pz[i, z[i,0]] for pz, z in zip(pzs, zs)]
                encf = "".join(chr(ord('a') + e) for e in enc)
                word = self.dataset.get_word(i)
                w.writerow([i, word, encf] + ps)

    def decode_sample(self, x, y):
        word = self.dataset.get_word(x)
        ctx = [self.dataset.get_word(y[i]) for i in range(y.shape[0])]
        lctx = ctx[:self.window]
        rctx = ctx[self.window:]
        return "{} [{}] {}".format(" ".join(lctx), word, " ".join(rctx))

    def write_predictions(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        n = 128
        samples = 8
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word"] + ["Sample {}".format(i) for i in range(samples)])
            x = np.random.random_integers(0, self.dataset.k, size=(n,1))
            ys = [self.predict_model.predict(x, verbose=0) for _ in range(samples)]
            for i in range(n):
                ix = x[i,0]
                word = self.dataset.get_word(ix)
                samples = [self.decode_sample(i, y[i, :]) for y in ys]
                w.writerow([i, word] + samples)

    def on_epoch_end(self, epoch, logs, output_path):
        if (epoch+1)%10 == 0:
            self.write_predictions("{}/predictions-{:08d}.csv".format(output_path, epoch))
            self.write_encodings("{}/predictions-{:08d}.csv".format(output_path, epoch))
            self.model.save_weights("{}/model-{:08d}.h5".format(output_path, epoch))

    def train(self, batch_size, epochs, steps_per_epoch, output_path, **kwargs):
        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)
        def on_epoch_end(epoch, logs):
            self.on_epoch_end(epoch, logs, output_path)
        cb = LambdaCallback(on_epoch_end=on_epoch_end)
        csvcb = CSVLogger("{}/history.csv".format(output_path))

        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                 callbacks=[cb, csvcb],
                                 **kwargs)
