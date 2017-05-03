"""
Train on given discrete sequential embeddings.
"""

import csv
import os

import numpy as np
from keras.callbacks import LambdaCallback, CSVLogger
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .util import latest_model
from ..layers.discrete_lstm import DiscreteLSTM
from ..layers.ngram_layer import NgramLayerGenerator
from ..layers.ngram_layer_distributed import NgramLayerDistributed
from ..layers.sequential_embedding_discrete import SequentialEmbedding


class WordSkipgramTest(object):
    def __init__(self, dataset, units, window, embedding, z_k, kernel_regularizer=None,
                 lr=1e-4):
        self.dataset = dataset
        self.units = units
        self.window = window
        self.embedding = embedding
        self.z_depth = embedding.shape[1]
        self.z_k = z_k
        print "Embedding type: {}".format(embedding.dtype)
        assert np.max(embedding, axis=None) < z_k
        assert np.min(embedding, axis=None) >= 0
        self.y_depth = window * 2
        k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((self.y_depth,), dtype='int32', name='input_y')

        elayer = SequentialEmbedding(embedding)
        z = elayer(input_x)
        print "Z type: {}".format(z.dtype)
        hlstm = DiscreteLSTM(k=z_k, units=units, kernel_regularizer=kernel_regularizer)
        zh = hlstm(z)
        skipgram = NgramLayerDistributed(k=k, units=units, kernel_regularizer=kernel_regularizer)
        nll = skipgram([zh, input_y])
        loss = Lambda(lambda _x: T.sum(_x, axis=1, keepdims=True), output_shape=lambda _x: (_x[0], 1))(nll)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def nll_initial(ytrue, ypred):
            return T.mean(nll[:, 0], axis=None)

        def nll_final(ytrue, ypred):
            return T.mean(nll[:, -1], axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[loss])
        self.model.compile(opt, loss_f, metrics=[nll_initial, nll_final])

        self.model_encode = Model(inputs=[input_x], outputs=[z])

        srng = RandomStreams(123)
        policy = NgramLayerGenerator(skipgram, srng=srng, depth=self.y_depth)
        zhfinal = Lambda(lambda _x: _x[:, -1, :], output_shape=lambda _x: (_x[0], _x[2]))(zh)
        ypred = policy(zhfinal)
        self.model_predict = Model(inputs=[input_x], outputs=[ypred])

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
        n = 128
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word"] + ["Sample {}".format(i) for i in range(samples)])
            x = np.random.randint(0, self.dataset.k, size=(n, 1))
            ys = [self.model_predict.predict(x, verbose=0) for _ in range(samples)]
            for i in range(n):
                ix = x[i, 0]
                word = self.dataset.get_word(ix)
                samples = [self.decode_sample(ix, y[i, :]) for y in ys]
                w.writerow([ix, word] + samples)

    def on_epoch_end(self, output_path, frequency, epoch, logs):
        if (epoch + 1) % frequency == 0:
            self.write_predictions("{}/predictions-{:08d}.csv".format(output_path, epoch))
            self.model.save_weights("{}/model-{:08d}.h5".format(output_path, epoch))

    def continue_training(self, output_path):
        initial_epoch = 0
        ret = latest_model(output_path, "model-(\\d+).h5")
        if ret:
            self.model.load_weights(ret[0])
            initial_epoch = ret[1] + 1
            print "Resuming training at {}".format(initial_epoch)
        return initial_epoch

    def train(self, batch_size, epochs, steps_per_epoch, output_path, frequency=10, continue_training=True, **kwargs):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        initial_epoch = 0
        if continue_training:
            initial_epoch = self.continue_training(output_path)

        def on_epoch_end(epoch, logs):
            self.on_epoch_end(output_path, frequency, epoch, logs)

        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)
        csvcb = CSVLogger("{}/history.csv".format(output_path), append=continue_training)
        cb = LambdaCallback(on_epoch_end=on_epoch_end)
        self.model.fit_generator(gen, epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=[cb, csvcb],
                                 initial_epoch=initial_epoch,
                                 **kwargs)
