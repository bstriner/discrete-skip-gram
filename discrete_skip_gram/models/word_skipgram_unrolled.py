import numpy as np
import theano
from keras.layers import Input, Embedding, Lambda, Add
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..layers.unrolled.bias_layer import BiasLayer
from ..layers.unrolled.decoder_layer import DecoderLayer
from ..layers.unrolled.encoder_layer import EncoderLayer
from ..layers.unrolled.sampler_layer import SamplerLayer
from ..layers.unrolled.skipgram_layer import SkipgramLayer, SkipgramPolicyLayer
from ..layers.utils import drop_dim_2, zeros_layer


class WordSkipgramUnrolled(object):
    def __init__(self, dataset, units, window, z_depth, z_k, schedule,
                 lr=1e-4,
                 act_reg=None,
                 reg=None):
        self.dataset = dataset
        self.units = units
        self.window = window
        self.z_depth = z_depth
        self.z_k = z_k
        self.y_depth = window * 2
        assert (len(schedule.shape) == 1)
        assert (schedule.shape[0] == z_depth)
        self.schedule = theano.shared(np.float32(schedule), name="schedule")
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
        encoder_h0 = BiasLayer(units)(input_x)
        pzs = []
        zs = []
        ht = encoder_h0
        zt = zeros_layer(1, dtype='int32')(input_x)
        encoder_layer = EncoderLayer(units=units, z_k=z_k, kernel_regularizer=reg)
        sampler = SamplerLayer(srng, offset=1)
        for i in range(z_depth):
            # print "Depth: {}".format(i)
            ht, pz = encoder_layer([ht, zt, embedded_x])
            zt = sampler(pz)
            pzs.append(pz)
            # print "PZ shape: {}".format(pz._keras_shape)
            zs.append(Lambda(lambda _z: _z - 1, output_shape=lambda _z: _z)(zt))

        # decode to zh
        zhs = []
        losses = []
        decoder_h0 = BiasLayer(units)
        ht = decoder_h0(input_x)
        decoder_layer = DecoderLayer(units, z_k=z_k, kernel_regularizer=reg)
        skipgram_layer = SkipgramLayer(units=units, k=x_k, kernel_regularizer=reg)
        for i, z in enumerate(zs):
            zlosses = []
            for zind in range(z_k):
                ztmp = Lambda(lambda _x: T.ones(_x.shape, dtype='int32') * (zind), output_shape=lambda _x: _x)(z)
                _, zh = decoder_layer([ht, ztmp])
                nll = skipgram_layer([zh, input_y])
                loss = Lambda(lambda (_a, _b): _a * (_b[:, zind].dimshuffle((0, 'x'))),
                              output_shape=lambda (_a, _b): _a)([nll, pzs[i]])

                zlosses.append(loss)
                """
                print "Loss: {}".format(loss._keras_shape)       
                _n = 128
                _x = np.random.randint(low=0, high=x_k, size=(_n, 1))
                _y = np.random.randint(low=0, high=x_k, size=(_n, self.y_depth))
                tm = Model([input_x, input_y], loss)
                _loss =tm.predict([_x, _y])
                print "_loss Shape: {} / {}".format(_loss.shape, loss._keras_shape)
                tm = Model([input_x, input_y], nll)
                _nll =tm.predict([_x, _y])
                print "_nll Shape: {} / {}".format(_nll.shape, nll._keras_shape)
                tm = Model([input_x, input_y], pzs[i])
                _nll =tm.predict([_x, _y])
                print "pzs[i] Shape: {} / {}".format(_nll.shape, pzs[i]._keras_shape)
                """
            zloss = Add()(zlosses)
            losses.append(zloss)

            ht, zh = decoder_layer([ht, z])
            zhs.append(zh)

        #nllconcat = Concatenate()(losses)
        # print "NLLCONCAT: {}".format(nllconcat._keras_shape)
        # nlltot = Lambda(lambda _nll: T.sum(_nll * (self.schedule.dimshuffle(('x', 0))), axis=1, keepdims=True),
        #                output_shape=lambda _nll: (_nll[0], 1))(nllconcat)
        nlltot = Add()(losses)

        def nll_initial(ytrue, ypred):
            return T.mean(losses[0], axis=None)

        def nll_final(ytrue, ypred):
            return T.mean(losses[-1], axis=None)

        self.model = Model([input_x, input_y], nlltot)
        opt = Adam(lr)
        self.model.compile(opt, lambda ytrue, ypred: ypred, metrics=[nll_initial, nll_final])

        """
        _n = 128
        _x = np.random.randint(low=0, high=x_k, size=(_n, 1))
        _y = np.random.randint(low=0, high=x_k, size=(_n, self.y_depth))
        _nll = self.model.predict([_x, _y])
        print "NLL Shape: {} / {}".format(_nll.shape, nlltot._keras_shape)
        """
        # Prediction model
        policy_layer = SkipgramPolicyLayer(skipgram_layer, srng=srng, depth=self.y_depth)
        ygen = policy_layer(zhs[-1])
        self.predict_model = Model([input_x], ygen)

        # Encoder model
        sampler_det = Lambda(lambda _p: T.argmax(_p, axis=1, keepdims=True) + 1,
                             output_shape=lambda _p: (_p[0], 1))
        zdets = []
        ht = encoder_h0
        zt = zeros_layer(1, dtype='int32')(input_x)
        for i in range(z_depth):
            ht, pz = encoder_layer([ht, zt, embedded_x])
            zt = sampler_det(pz)
            zdets.append(Lambda(lambda _z: _z - 1, output_shape=lambda _z: _z)(zt))
        self.encode_model = Model([input_x], zdets)

        # Decoder model
        input_zs = [Input((1,), dtype='int32', name="input_z_{}".format(i)) for i in range(z_depth)]

        ht = decoder_h0(input_zs[0])
        #        decoder_layer = DecoderLayer(units, z_k=z_k, kernel_regularizer=reg)
        #        skipgram_layer = SkipgramLayer(units=units, k=x_k, kernel_regularizer=reg)
        #        policy_layer = SkipgramPolicyLayer(skipgram_layer, srng=srng, depth=self.y_depth)

        for z in input_zs:
            ht, zhdec = decoder_layer([ht, z])

        ygen = policy_layer(zhdec)
        self.decoder_model = Model(input_zs, ygen)

    def train(self, batch_size, epochs, steps_per_epoch, **kwargs):
        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)
