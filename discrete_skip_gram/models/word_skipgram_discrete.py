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
from ..layers.highway_layer import HighwayLayer
from ..layers.time_distributed_dense import TimeDistributedDense
from ..layers.utils import drop_dim_2, zeros_layer, add_layer, nll_metrics
from ..layers.adversary_layer import AdversaryLayer
from .util import latest_model

from discrete_skip_gram.layers.utils import leaky_relu
from ..layers.utils import softmax_nd_layer, shift_tensor_layer, softmax_nd
from ..layers.shift_padding_layer import ShiftPaddingLayer
from ..layers.highway_layer_discrete import HighwayLayerDiscrete
from ..layers.skipgram_layer_discrete import SkipgramLayerDiscrete, SkipgramPolicyLayerDiscrete
from .sg_model import SGModel


def selection_layer(zind):
    return Lambda(lambda (_a, _b): _a * (_b[:, zind].dimshuffle((0, 'x'))), output_shape=lambda (_a, _b): _a)


class WordSkipgramDiscrete(SGModel):
    def __init__(self, dataset, units, embedding_units, window, z_depth, z_k,
                 lr=1e-4,
                 lr_a=1e-3,
                 inner_activation=leaky_relu,
                 kernel_regularizer=None,
                 embeddings_regularizer=None,
                 hidden_layers=2,
                 layernorm=False,
                 adversary_weight=1.0
                 ):
        self.dataset = dataset
        self.units = units
        self.embedding_units = embedding_units
        self.window = window
        self.hidden_layers = hidden_layers
        self.z_depth = z_depth
        self.z_k = z_k
        self.y_depth = window * 2
        self.inner_activation = inner_activation
        assert z_depth > 0
        srng = RandomStreams(123)
        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((self.y_depth,), dtype='int32', name='input_y')

        x_embedding = Embedding(input_dim=self.dataset.k,
                                output_dim=z_depth * z_k,
                                embeddings_regularizer=embeddings_regularizer)
        rs = Reshape((z_depth, z_k))
        sm = softmax_nd_layer()
        z = sm(rs(x_embedding(input_x)))  # n, z_depth, z_k

        sampler = Lambda(lambda _x: T.argmax(_x, axis=2), output_shape=lambda _x: (_x[0], _x[1]), name='z_sampler')
        z_sampled = sampler(z)
        z_shifted = shift_tensor_layer()(z_sampled)

        zrnn = HighwayLayerDiscrete(units=self.units,
                                    embedding_units=self.embedding_units,
                                    k=self.z_k + 1,
                                    layernorm=layernorm,
                                    inner_activation=self.inner_activation,
                                    hidden_layers=self.hidden_layers,
                                    kernel_regularizer=kernel_regularizer)

        zh = zrnn(z_shifted)  # n, z_depth, units

        skipgram = SkipgramLayerDiscrete(units=self.units,
                                         embedding_units=self.embedding_units,
                                         z_k=z_k,
                                         y_k=x_k,
                                         layernorm=layernorm,
                                         inner_activation=self.inner_activation,
                                         hidden_layers=self.hidden_layers,
                                         embeddings_regularizer=embeddings_regularizer,
                                         kernel_regularizer=kernel_regularizer)

        nll = skipgram([zh, input_y])  # n, z_depth, z_k

        loss_layer = Lambda(lambda (_a, _b): T.sum(T.sum(_a * _b, axis=2), axis=1, keepdims=True),
                            output_shape=lambda (_a, _b): (_a[0], 1),
                            name="loss_layer")
        loss = loss_layer([nll, z])

        # adversary: try to minimize kl with z
        input_z = Input((self.z_depth, self.z_k), dtype='float32', name='input_z')
        z_shift = ShiftPaddingLayer()
        arnn = HighwayLayer(units=units,
                            inner_activation=inner_activation,
                            hidden_layers=hidden_layers,
                            kernel_regularizer=kernel_regularizer)
        ah = z_shift(input_z)
        ah = arnn(ah)
        ah = TimeDistributedDense(units,
                                  activation=inner_activation,
                                  kernel_regularizer=kernel_regularizer,
                                  name="adversary_d1")(ah)
        ah = TimeDistributedDense(units,
                                  activation=inner_activation,
                                  kernel_regularizer=kernel_regularizer,
                                  name="adversary_d2")(ah)
        d = TimeDistributedDense(z_k,
                                 activation=softmax_nd,
                                 kernel_regularizer=kernel_regularizer,
                                 name="adversary_d3")(ah)
        adversary = Model(inputs=[input_z], outputs=[d])

        # train adversary
        dz = adversary(z)

        def em(yt, yp):
            return T.mean(T.sum(T.sum(T.abs_(yt - yp), axis=2), axis=1), axis=0)

        aem = em(z, dz)
        adversary_reg = 0
        for layer in adversary.layers:
            for l in layer.losses:
                adversary_reg += l
        aopt = Adam(lr=lr_a)
        atotal = aem + adversary_reg
        aweights = adversary.trainable_weights
        print "A weights: {}".format(aweights)
        aupdates = aopt.get_updates(aweights, {}, atotal)
        self.adversary_weight = K.variable(np.float32(adversary_weight), dtype='float32', name='adversary_weight')
        regloss = -self.adversary_weight * aem

        self.adversary = adversary
        skipgram.add_loss(regloss)
        skipgram.add_update(aupdates)

        self.model = Model([input_x, input_y], loss)

        def adversary_em(ytrue, ypred):
            return aem

        def adversary_loss(ytrue, ypred):
            return atotal

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        avg_nll = T.mean(nll, axis=0)
        metrics = nll_metrics(avg_nll, z_depth)
        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[loss])
        self.model.compile(opt, loss_f, metrics=[adversary_loss, adversary_em] + metrics)
        self.weights = self.model.weights + self.adversary.weights + opt.weights + aopt.weights

        # Prediction model
        policy_layer = SkipgramPolicyLayerDiscrete(skipgram, srng=srng, depth=self.y_depth)
        zhfinal = Lambda(lambda _z: _z[:, -1, :], output_shape=lambda _z: (_z[0], _z[2]), name="zhfinal")(zh)
        zfinal = Lambda(lambda _z: _z[:, -1:], output_shape=lambda _z: (_z[0], 1), name="zfinal")(z_sampled)
        ygen = policy_layer([zhfinal, zfinal])
        self.model_predict = Model([input_x], ygen)

        # Encoder model
        self.model_encode = Model([input_x], z_sampled)

    def write_encodings(self, output_path):
        x = np.arange(self.dataset.k).reshape((-1, 1))
        z = self.model_encode.predict(x, verbose=0)

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path + ".csv", 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word", "Encoding"] + ["Cat {}".format(j) for j in range(self.z_depth)])
            for i in range(self.dataset.k):
                enc = z[i, :]
                enca = [enc[j] for j in range(enc.shape[0])]
                encf = "".join(chr(ord('a') + e) for e in enca)
                word = self.dataset.get_word(i)
                w.writerow([i, word, encf] + enca)
        np.save(output_path + ".npy", z)

    def on_epoch_end(self, output_path, epoch):
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.write_predictions_flat("{}/predictions-{:08d}.csv".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
