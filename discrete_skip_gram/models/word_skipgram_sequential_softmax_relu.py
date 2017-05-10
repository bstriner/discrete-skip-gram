import csv
import os

import numpy as np
from keras import backend as K
from keras.layers import Input, Embedding, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..layers.unrolled.skipgram_layer_distributed_relu import SkipgramLayerDistributedRelu
from ..layers.unrolled.skipgram_layer_distributed_relu import SkipgramPolicyDistributedLayerRelu
from ..layers.time_distributed_dense import TimeDistributedDense
from ..layers.utils import softmax_nd_layer, softmax_nd, nll_metrics
from .sg_model import SGModel
from ..layers.shift_padding_layer import ShiftPaddingLayer
from ..layers.highway_layer import HighwayLayer


class WordSkipgramSequentialSoftmaxRelu(SGModel):
    def __init__(self,
                 dataset,
                 units,
                 z_depth,
                 z_k,
                 schedule,
                 window,
                 adversary_weight,
                 embedding_units,
                 hidden_layers=1,
                 inner_activation=T.nnet.relu,
                 kernel_regularizer=None,
                 lr=1e-4, lr_a=3e-4):
        self.dataset = dataset
        self.units = units
        self.z_depth = z_depth
        self.z_k = z_k
        self.window = window
        self.srng = RandomStreams(123)
        assert (len(schedule.shape) == 1)
        assert (schedule.shape[0] == z_depth)

        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((window * 2,), dtype='int32', name='input_y')

        # encoder
        embedding = Embedding(x_k, z_depth * z_k, name="x_embedding")
        rs = Reshape((z_depth, z_k), name="z_reshape")
        sm = softmax_nd_layer(name="z_softmax")
        z = sm(rs(embedding(input_x)))  # n, z_depth, z_k

        # z hidden
        highway = HighwayLayer(units=units,
                               inner_activation=inner_activation,
                               hidden_layers=hidden_layers,
                               kernel_regularizer=kernel_regularizer,
                               name="z_hidden_rnn")
        zh = highway(z)

        # decoder
        ngram = SkipgramLayerDistributedRelu(k=x_k,
                                             units=units,
                                             kernel_regularizer=kernel_regularizer,
                                             embedding_units=embedding_units,
                                             hidden_layers=hidden_layers,
                                             inner_activation=inner_activation,
                                             name="skipgram")
        nll = ngram([zh, input_y])

        # loss calculation
        nll_weights = K.variable(schedule, name="schedule", dtype='float32')
        nll_weighted_loss = Lambda(lambda _nll: T.sum(_nll * (nll_weights.dimshuffle(('x', 0))), axis=1, keepdims=True),
                                   output_shape=lambda _nll: (_nll[0], 1), name="weighted_loss")(nll)

        # adversary: try to minimize kl with z
        input_z = Input((self.z_depth, self.z_k), dtype='float32', name='input_z')
        z_shift = ShiftPaddingLayer()
        arnn = HighwayLayer(units=units,
                            inner_activation=inner_activation,
                            hidden_layers=hidden_layers,
                            kernel_regularizer=kernel_regularizer,
                            )
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

        def adversary_em(ytrue, ypred):
            return aem

        def adversary_loss(ytrue, ypred):
            return atotal

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        avg_nll = T.mean(nll, axis=0)
        metrics = nll_metrics(avg_nll, z_depth)

        opt = Adam(lr)

        ngram.add_loss(regloss)
        ngram.add_update(updates=aupdates)

        self.model = Model(inputs=[input_x, input_y], outputs=[nll_weighted_loss])
        self.model.compile(opt, loss_f, metrics=[adversary_loss, adversary_em] + metrics)
        self.weights = self.model.weights + self.adversary.weights + opt.weights + aopt.weights

        # Encoder
        self.model_encode = Model(inputs=[input_x], outputs=[z])

        # Prediction model
        policy = SkipgramPolicyDistributedLayerRelu(ngram, srng=self.srng, depth=self.window * 2)
        zhfinal = Lambda(lambda _z: _z[:, -1, :], output_shape=lambda _z: (_z[0], _z[2]))(zh)
        ypred = policy(zhfinal)
        self.model_predict = Model(inputs=[input_x], outputs=[ypred])

    def summary(self):
        print "Main model"
        self.model.summary()
        print "Adverary model"
        self.adversary.summary()

    def write_encodings(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path + ".csv", 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word", "Encoding"] + ["P{}".format(j) for j in range(self.z_depth)])
            x = np.arange(self.dataset.k).reshape((-1, 1))
            z = self.model_encode.predict(x, verbose=0)
            for i in range(self.dataset.k):
                enc = z[i, :, :]
                t = np.argmax(enc, axis=1)
                ps = np.max(enc, axis=1)
                encf = "".join(chr(ord('a') + t[j]) for j in range(t.shape[0]))
                psf = [ps[j] for j in range(self.z_depth)]
                word = self.dataset.get_word(i)
                w.writerow([i, word, encf] + psf)
            encodings = np.argmax(z, axis=2)
            np.save(output_path + ".npy", encodings)

    def on_epoch_end(self, output_path, epoch):
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.write_predictions_flat("{}/predictions-{:08d}.csv".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
