import csv
import os

import numpy as np
from keras.callbacks import LambdaCallback, CSVLogger
from keras.layers import Input, Embedding
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .sg_model import SGModel
from .util import latest_model
from ..hsm import hsm_decode
from ..layers.sequential_embedding_discrete import SequentialEmbeddingDiscrete
from ..layers.skipgram_hsm_layer import SkipgramHSMLayer, SkipgramHSMPolicyLayer
from ..layers.utils import drop_dim_2


class WordSkipgramBaselineHSM(SGModel):
    def __init__(self, dataset, units, window,
                 hsm,
                 kernel_regularizer=None,
                 lr=1e-4):
        self.dataset = dataset
        self.units = units
        self.window = window
        self.y_depth = window * 2
        self.codes, self.wordcodes = hsm

        k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((self.y_depth,), dtype='int32', name='input_y')

        x_embedding = Embedding(k, units, embeddings_regularizer=kernel_regularizer)
        z = drop_dim_2()(x_embedding(input_x))
        skipgram = SkipgramHSMLayer(units=units,
                                    kernel_regularizer=kernel_regularizer,
                                    embeddings_regularizer=kernel_regularizer)

        y_embedding = SequentialEmbeddingDiscrete(self.codes)
        y_embedded = y_embedding(input_y)

        nll = skipgram([z, y_embedded])

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def avg_nll(ytrue, ypred):
            return T.mean(nll, axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[nll])
        self.model.compile(opt, loss_f, metrics=[avg_nll])
        self.weights = self.model.weights + opt.weights

        self.model_encode = Model(inputs=[input_x], outputs=[z])

        srng = RandomStreams(123)
        policy = SkipgramHSMPolicyLayer(skipgram, srng=srng, y_depth=self.y_depth, code_depth=self.codes.shape[1])
        ypred = policy(z)
        self.model_predict = Model(inputs=[input_x], outputs=[ypred])

    def summary(self):
        self.model.summary()

    def write_encodings(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        x = np.arange(self.dataset.k).reshape((-1, 1))
        z = self.model_encode.predict(x, verbose=0)
        np.save(output_path + ".npy", z)

    def decode_sample(self, x, y):
        word = self.dataset.get_word(x)
        ctx = [self.dataset.get_word(hsm_decode(y[i, :], self.codes)) for i in range(y.shape[0])]
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
                samples = [self.decode_sample(ix, y[i, :, :]) for y in ys]
                w.writerow([ix, word] + samples)

    def on_epoch_end(self, output_path, epoch):
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.write_predictions("{}/predictions-{:08d}.csv".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))

    def continue_training(self, output_path):
        initial_epoch = 0
        ret = latest_model(output_path, "model-(\\d+).h5")
        if ret:
            self.load(ret[0])
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
            if (epoch + 1) % frequency == 0:
                self.on_epoch_end(output_path, epoch)

        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)
        csvcb = CSVLogger("{}/history.csv".format(output_path), append=continue_training)
        cb = LambdaCallback(on_epoch_end=on_epoch_end)
        self.model.fit_generator(gen, epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=[cb, csvcb],
                                 initial_epoch=initial_epoch,
                                 **kwargs)
