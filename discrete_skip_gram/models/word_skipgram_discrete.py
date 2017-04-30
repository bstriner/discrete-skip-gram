"""
Each element of sequence is an embedding layer
"""
import keras.backend as K
import csv
import os
import numpy as np
import theano
from keras.callbacks import LambdaCallback, CSVLogger
from keras.layers import Input, Embedding, Lambda, Activation, Add, Concatenate, Reshape
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..layers.unrolled.bias_layer import BiasLayer
from ..layers.unrolled.decoder_layer import DecoderLayer
from ..layers.unrolled.sampler_deterministic_layer import SamplerDeterministicLayer
from ..layers.unrolled.skipgram_batch_layer import SkipgramBatchLayer, SkipgramBatchPolicyLayer
from ..layers.utils import drop_dim_2, zeros_layer, add_layer
from ..layers.adversary_layer import AdversaryLayer
from .util import latest_model

def selection_layer(zind):
    return Lambda(lambda (_a, _b): _a * (_b[:, zind].dimshuffle((0, 'x'))), output_shape=lambda (_a, _b): _a)


class WordSkipgramDiscrete(object):
    def __init__(self, dataset, units, window, z_depth, z_k,
                 lr=1e-4,
                 train_rate=5,
                 kernel_regularizer=None,
                 adversary_weight=1.0
                 ):
        self.train_rate=train_rate
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

        decoder_layer = DecoderLayer(units, z_k=z_k + 1, kernel_regularizer=kernel_regularizer,
                                     name="decoder")
        skipgram_layer = SkipgramBatchLayer(units=units, y_k=x_k, z_k=z_k, kernel_regularizer=kernel_regularizer,
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
        zvec = Concatenate()(zs)
        pzvec = Concatenate(axis=1)([Reshape((1,z_k))(pz) for pz in pzs])
        adversary = AdversaryLayer(z_k=z_k, units=units, kernel_regularizer=kernel_regularizer)
        ay = adversary(zvec)
        aloss = T.mean(T.abs_(ay-pzvec), axis=None)
        areg = -adversary_weight*aloss
        alossreg = 0
        for l in adversary.losses:
            alossreg += l
        aopt = Adam(lr)
        aupdates = aopt.get_updates(adversary.weights, {}, aloss+alossreg)

        opt = Adam(lr)
        if self.z_depth == 1:
            loss = losses[0]
        else:
            loss = Add()(losses)
        skipgram_layer.add_loss(areg)
        self.model = Model([input_x, input_y], loss)

        def nll_initial(_yt, _yp):
            return T.mean(losses[0], axis=None)

        def nll_final(_yt, _yp):
            return T.mean(losses[-1], axis=None)

        def adversary_loss(_yt, _yp):
            return T.mean(aloss, axis=None)

        self.model.compile(opt, loss_f, metrics=[nll_initial, nll_final, adversary_loss])
        self.model._make_train_function()

        inputs = self.model._feed_inputs + self.model._feed_targets + self.model._feed_sample_weights
        outputs = [self.model.total_loss] + self.model.metrics_tensors
        trainf = self.model.train_function
        adversary_function = K.function(inputs, outputs, updates=aupdates)

        self.counter = 0

        def train_function(inputs):
            self.counter += 1
            if self.counter > self.train_rate + 1:
                self.counter = 0
            if self.counter > 0:
                return adversary_function(inputs)
            else:
                return trainf(inputs)

        self.model.train_function = train_function

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
        self.model_combined = Model([input_x, input_y]+ input_zs,[loss, ay])

    def write_generated(self, output_path):
        n = 128
        samples = 8
        _, x = self.dataset.cbow_batch(n=n, window=self.window, test=True)
        ys = [self.predict_model.predict(x, verbose=0) for _ in range(samples)]
        with open(output_path, 'w') as f:
            for i in range(n):
                strs = []
                w = self.dataset.get_word(x[i, 0])
                for y in ys:
                    ctx = [self.dataset.get_word(y[i, j]) for j in range(self.window * 2)]
                    lctx = " ".join(ctx[:self.window])
                    rctx = " ".join(ctx[self.window:])
                    strs.append("{} [{}] {}".format(lctx, w, rctx))
                f.write("{}: {}\n".format(w, " | ".join(strs)))

    def write_encoded(self, output_path):
        x = np.arange(self.dataset.k).reshape((-1, 1))
        ret = self.encode_model.predict(x, verbose=0)
        pzs, zs = ret[:self.z_depth], ret[self.z_depth:]

        # if z_depth == 1:
        #    zs = [zs]
        with open(output_path, 'wb') as f:
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

    def continue_training(self, output_path):
        initial_epoch = 0
        ret = latest_model(output_path, "model-(\\d+).h5")
        if ret:
            self.model_combined.load_weights(ret[0])
            initial_epoch = ret[1] + 1
        print "Resuming training at {}".format(initial_epoch)
        return initial_epoch

    def train(self, batch_size, epochs, steps_per_epoch, output_path, continue_training=True,
              frequency=10, **kwargs):
        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        initial_epoch = 0
        if continue_training:
            initial_epoch = self.continue_training(output_path)
        def on_epoch_end(epoch, logs):
            if (epoch + 1) % frequency == 0:
                self.write_generated("{}/generated-{:08d}.txt".format(output_path, epoch))
                self.write_encoded("{}/encoded-{:08d}.csv".format(output_path, epoch))
                self.model_combined.save_weights("{}/model-{:08d}.h5".format(output_path, epoch))

        csvpath = "{}/history.csv".format(output_path)
        cbs = [LambdaCallback(on_epoch_begin=on_epoch_end), CSVLogger(csvpath, append=continue_training)]
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=cbs,
                                 initial_epoch=initial_epoch,
                                 verbose=1, **kwargs)
