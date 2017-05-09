import csv
import os

import numpy as np
from keras import backend as K
from keras.callbacks import LambdaCallback, CSVLogger
from keras.layers import Input, Embedding, Lambda, Activation
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..layers.encoder_lstm_continuous import EncoderLSTMContinuous
from ..layers.unrolled.skipgram_layer_distributed_relu import SkipgramLayerDistributedRelu
from ..layers.unrolled.skipgram_layer_distributed_relu import SkipgramPolicyDistributedLayerRelu
from ..layers.time_distributed_dense import TimeDistributedDense
from ..layers.utils import drop_dim_2, softmax_nd_layer, softmax_nd
from .sg_model import SGModel


class WordSkipgramSequentialSigmoid(SGModel):
    def __init__(self, dataset, units, z_depth, schedule, window, adversary_weight,
                 embedding_units,
                 hidden_layers=1,
                 inner_activation=T.nnet.relu,
                 kernel_regularizer=None,
                 lr=1e-4, lr_a=3e-4):
        self.dataset = dataset
        self.units = units
        self.z_depth = z_depth
        self.window = window
        self.srng = RandomStreams(123)
        assert (len(schedule.shape) == 1)
        assert (schedule.shape[0] == z_depth)

        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((window * 2,), dtype='int32', name='input_y')

        # encoder
        embedding = Embedding(x_k, embedding_units)
        embedded_x = drop_dim_2()(embedding(input_x))
        encoder = EncoderLSTMContinuous(z_depth, z_k=1, units=units, activation='sigmoid',
                                        kernel_regularizer=kernel_regularizer)
        z = encoder(embedded_x)  # n, z_depth, z_k

        #z hidden
        lstm = LSTM(units, return_sequences=True, kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=kernel_regularizer)
        zd1 = TimeDistributedDense(units, activation='tanh', kernel_regularizer=kernel_regularizer)
        zd2 = TimeDistributedDense(units, kernel_regularizer=kernel_regularizer)
        zh = zd2(zd1(lstm(z)))

        # decoder
        ngram = SkipgramLayerDistributedRelu(k=x_k, units=units, kernel_regularizer=kernel_regularizer,
                                             embedding_units=embedding_units,
                                             hidden_layers=hidden_layers,
                                             inner_activation=inner_activation)

        nll = ngram([zh, input_y])

        # loss calculation
        nll_weights = K.variable(schedule, name="schedule", dtype='float32')
        nll_weighted_loss = Lambda(lambda _nll: T.sum(_nll * (nll_weights.dimshuffle(('x', 0))), axis=1, keepdims=True),
                                   output_shape=lambda _nll: (_nll[0], 1))(nll)

        # adversary: try to minimize kl with z
        input_z = Input((self.z_depth, 1), dtype='float32', name='input_z')
        z_shift = Lambda(lambda _z: T.concatenate((T.zeros_like(_z[:, 0:1, :]), _z[:, :-1, :]), axis=1),
                         output_shape=lambda _z: _z)
        alstm = LSTM(units, kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                     return_sequences=True)
        ad1 = TimeDistributedDense(units, activation='tanh', kernel_regularizer=kernel_regularizer)
        ad2 = TimeDistributedDense(1, activation='sigmoid', kernel_regularizer=kernel_regularizer)
        h = z_shift(input_z)
        h = alstm(h)
        h = ad1(h)
        d = ad2(h)
        adversary = Model(inputs=[input_z], outputs=[d])

        # train adversary
        dz = adversary(z)

        def em(yt, yp):
            return T.mean(T.sum(T.sum(T.abs_(yt - yp), axis=2), axis=1), axis=0)

        aloss = em(z, dz)
        adversary_reg = 0
        for layer in adversary.layers:
            for l in layer.losses:
                print "Layer loss: {}".format(layer)
                adversary_reg += l
        aopt = Adam(lr=lr_a)
        aupdates = aopt.get_updates(adversary.trainable_weights, {}, aloss + adversary_reg)
        self.adversary_weight = K.variable(np.float32(adversary_weight), dtype='float32', name='adversary_weight')
        regloss = -self.adversary_weight * aloss

        self.adversary = adversary

        def adversary_loss(ytrue, ypred):
            return aloss

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def nll_initial(ytrue, ypred):
            return T.mean(nll[:, 0], axis=None)

        def nll_final(ytrue, ypred):
            return T.mean(nll[:, -1], axis=None)

        opt = Adam(lr)

        ngram.add_loss(regloss)
        ngram.add_update(updates=aupdates)

        self.model = Model(inputs=[input_x, input_y], outputs=[nll_weighted_loss])
        self.model.compile(opt, loss_f, metrics=[nll_initial, nll_final, adversary_loss])
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
