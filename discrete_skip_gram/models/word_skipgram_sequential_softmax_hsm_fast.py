import csv
import os

import keras.backend as K
import numpy as np
from keras.layers import Input, Embedding, Reshape, Lambda, Activation
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .sg_model import SGModel
from ..layers.sequential_embedding_discrete import SequentialEmbeddingDiscrete
from ..layers.skipgram_hsm_layer_fast import SkipgramHSMLayerFast, SkipgramHSMPolicyLayerFast
from ..layers.time_distributed_dense import TimeDistributedDense
from ..layers.utils import softmax_nd_layer


class WordSkipgramSequentialSoftmaxHSMFast(SGModel):
    def __init__(self, dataset, units, window,
                 z_k, z_depth,
                 schedule,
                 hsm,
                 kernel_regularizer=None,
                 lr=1e-4,
                 lr_a=1e-3,
                 adversary_weight=1.0):
        self.dataset = dataset
        self.units = units
        self.z_k = z_k
        self.z_depth = z_depth
        self.window = window
        self.y_depth = window * 2
        self.hsm = hsm

        k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((self.y_depth,), dtype='int32', name='input_y')

        x_embedding = Embedding(k, z_depth * z_k)  # no regularizer here
        rs = Reshape((z_depth, z_k))
        sm = softmax_nd_layer()
        z = sm(rs(x_embedding(input_x)))

        zlstm = LSTM(units, return_sequences=True,
                     recurrent_regularizer=kernel_regularizer,
                     kernel_regularizer=kernel_regularizer)
        zd1 = TimeDistributedDense(units, activation="tanh")
        zd2 = TimeDistributedDense(units)
        zh = zd2(zd1(zlstm(z)))

        skipgram = SkipgramHSMLayerFast(units=units,
                                        kernel_regularizer=kernel_regularizer,
                                        embeddings_regularizer=kernel_regularizer)

        y_embedding = SequentialEmbeddingDiscrete(self.hsm.codes)
        y_embedded = y_embedding(input_y)

        nll = skipgram([zh, y_embedded])  # (n, z_depth)
        # loss calculation
        nll_weights = K.variable(schedule, name="schedule", dtype='float32')
        nll_weighted_loss = Lambda(lambda _nll: T.sum(_nll * (nll_weights.dimshuffle(('x', 0))), axis=1, keepdims=True),
                                   output_shape=lambda _nll: (_nll[0], 1))(nll)  # (n, 1)

        # build adversary
        input_z = Input((self.z_depth, self.z_k), dtype='float32', name='input_z')
        z_shift = Lambda(lambda _z: T.concatenate((T.zeros_like(_z[:, 0:1, :]), _z[:, :-1, :]), axis=1),
                         output_shape=lambda _z: _z)
        alstm = LSTM(units, kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                     return_sequences=True)
        ad1 = TimeDistributedDense(units)
        ad2 = TimeDistributedDense(z_k)
        act = Activation('tanh')
        sm = softmax_nd_layer()
        h = z_shift(input_z)
        h = alstm(h)
        h = ad1(h)
        h = act(h)
        h = ad2(h)
        d = sm(h)
        adversary = Model(inputs=[input_z], outputs=[d])
        print "Adverary model"
        adversary.summary()

        dz = adversary(z)

        def em3d(yt, yp):
            assert yt.ndim == 3
            assert yp.ndim == 3
            return T.mean(T.sum(T.sum(T.abs_(yt - yp), axis=2), axis=1), axis=0)

        aloss = em3d(z, dz)
        alossreg = 0.0
        for l in adversary.losses:
            alossreg += l
        aopt = Adam(lr=lr_a)
        aupdates = aopt.get_updates(adversary.trainable_weights, {}, aloss + alossreg)
        self.adversary_weight = K.variable(np.float32(adversary_weight), dtype='float32', name='adversary_weight')
        regloss = -self.adversary_weight * aloss

        skipgram.add_update(aupdates)
        skipgram.add_loss(regloss)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def avg_nll(ytrue, ypred):
            return T.mean(nll, axis=1)

        def initial_nll(ytrue, ypred):
            return nll[:, 0]

        def final_nll(ytrue, ypred):
            return nll[:, -1]

        def adversary_loss(ytrue, ypred):
            return aloss

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[nll_weighted_loss])
        self.model.compile(opt, loss_f, metrics=[avg_nll, initial_nll, final_nll, adversary_loss])
        print "Model updates: {}, aupdates: {}".format(len(self.model.updates), len(aupdates))
        self.weights = self.model.weights + opt.weights + adversary.weights + aopt.weights

        self.model_encode = Model(inputs=[input_x], outputs=[z])

        srng = RandomStreams(123)
        policy = SkipgramHSMPolicyLayerFast(skipgram,
                                            srng=srng,
                                            y_depth=self.y_depth,
                                            code_depth=self.hsm.codes.shape[1])
        zlast = Lambda(lambda _x: _x[:, -1, :], output_shape=lambda _x: (_x[0], _x[2]))(zh)
        ypred = policy(zlast)
        self.model_predict = Model(inputs=[input_x], outputs=[ypred])

    def summary(self):
        print "Skipgram Model"
        self.model.summary()
        print "Skipgram Policy Model"
        self.model_predict.summary()

    def write_encodings(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        x = np.arange(self.dataset.k).reshape((-1, 1))
        z = self.model_encode.predict(x, verbose=0)  # n, z_depth, z_k
        np.save(output_path + "-raw.npy", z)
        zd = np.argmax(z, axis=2).astype(np.int32)
        np.save(output_path + ".npy", zd)
        with open(output_path + ".csv", 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word", "Encoding"] + ["Cat {}".format(i) for i in range(self.z_depth)])
            for idx in range(self.dataset.k):
                word = self.dataset.get_word(idx)
                enc = zd[i, :]
                encs = [enc[j] for j in range(enc.shape[0])]
                encf = "".join(chr(ord('a') + e) for e in encs)
                w.writerow([idx, word, encf] + encs)

    def on_epoch_end(self, output_path, epoch):
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.write_predictions("{}/predictions-{:08d}.csv".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
