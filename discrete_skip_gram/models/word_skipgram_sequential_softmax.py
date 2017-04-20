import csv
import os

import numpy as np
from keras import backend as K
from keras.callbacks import LambdaCallback, CSVLogger
from keras.layers import Input, Embedding, Lambda
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..layers.encoder_lstm_continuous import EncoderLSTMContinuous
from ..layers.ngram_layer import NgramLayerGenerator
from ..layers.ngram_layer_distributed import NgramLayerDistributed
from ..layers.utils import drop_dim_2


class WordSkipgramSequentialSofttmax(object):
    def __init__(self, dataset, units, z_depth, z_k, schedule, window, frequency=5,
                 kernel_regularizer=None,
                 lr=1e-4):
        self.frequency = frequency
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
        input_y = Input((window*2,), dtype='int32', name='input_y')

        # encoder
        embedding = Embedding(x_k, units, embeddings_regularizer=kernel_regularizer)
        embedded_x = drop_dim_2()(embedding(input_x))
        encoder = EncoderLSTMContinuous(z_depth, z_k, units, activation=T.nnet.softmax,
                                        kernel_regularizer=kernel_regularizer)
        z = encoder(embedded_x)  # n, z_depth, z_k

        # decoder
        lstm = LSTM(units, return_sequences=True, kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=kernel_regularizer)
        ngram = NgramLayerDistributed(k=x_k, units=units, kernel_regularizer=kernel_regularizer)
        zh = lstm(z)
        nll = ngram([zh, input_y])

        # loss calculation
        weights = K.variable(schedule, name="schedule", dtype='float32')
        loss = Lambda(lambda _nll: T.sum(_nll * (weights.dimshuffle(('x', 0))), axis=1, keepdims=True),
                      output_shape=lambda _nll: (_nll[0], 1))(nll)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def nll_initial(ytrue, ypred):
            return T.mean(nll[:, 0], axis=None)

        def nll_final(ytrue, ypred):
            return T.mean(nll[:, -1], axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[loss])
        self.model.compile(opt, loss_f, metrics=[nll_initial, nll_final])
        self.model.summary()
        # Encoder
        self.model_encode = Model(inputs=[input_x], outputs=[z])

        # Prediction model
        policy = NgramLayerGenerator(ngram, srng=self.srng, depth=self.window * 2)
        zhfinal = Lambda(lambda _z: _z[:, -1, :], output_shape=lambda _z: (_z[0], _z[2]))(zh)
        ypred = policy(zhfinal)
        self.model_predict = Model(inputs=[input_x], outputs=[ypred])

    def write_encodings(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word", "Encoding"])
            x = np.arange(self.dataset.k).reshape((-1, 1))
            z = self.model_encode.predict(x, verbose=0)
            for i in range(self.dataset.k):
                enc = z[i, :, :]
                t = np.argmax(enc, axis=1)
                encf = "".join(chr(ord('a') + t[j]) for j in range(t.shape[0]))
                word = self.dataset.get_word(i)
                w.writerow([i, word, encf])

    def decode_sample(self, x, y):
        word = self.dataset.get_word(x)
        ctx = [self.dataset.get_word(y[i]) for i in range(y.shape[0])]
        lctx = ctx[:self.window]
        rctx = ctx[self.window:]
        return "{} [{}] {}".format(" ".join(lctx), word, " ".join(rctx))

    def write_predictions(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        samples = 8
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word"] + ["Sample {}".format(i) for i in range(samples)])
            x = np.arange(self.dataset.k).reshape((-1, 1))
            ys = [self.model_predict.predict(x, verbose=0) for _ in range(samples)]
            for i in range(self.dataset.k):
                word = self.dataset.get_word(i)
                samples = [self.decode_sample(i, y[i, :]) for y in ys]
                w.writerow([i, word] + samples)

    def train(self, batch_size, epochs, steps_per_epoch, output_path, **kwargs):
        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        def on_epoch_end(epoch, logs):
            if (epoch + 1) % self.frequency == 0:
                self.write_encodings(output_path="{}/encoded-{:08d}.csv".format(output_path, epoch))
                self.write_predictions(output_path="{}/predicted-{:08d}.csv".format(output_path, epoch))
                self.model.save_weights("{}/model-{:08d}.h5".format(output_path, epoch))

        csvcb = CSVLogger("{}/history.csv".format(output_path))
        lcb = LambdaCallback(on_epoch_end=on_epoch_end)
        cbs = [csvcb, lcb]
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=cbs, **kwargs)
