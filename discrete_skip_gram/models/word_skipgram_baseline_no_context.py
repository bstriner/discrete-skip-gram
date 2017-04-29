import csv
import os

import numpy as np
from keras.callbacks import LambdaCallback, CSVLogger
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..layers.unrolled.skipgram_layer import SkipgramLayer, SkipgramPolicyLayer
from ..layers.utils import zeros_layer


class WordSkipgramBaselineNoContext(object):
    def __init__(self, dataset, units, window,
                 lr=1e-4):
        self.dataset = dataset
        self.units = units
        self.window = window
        self.y_depth = window * 2
        k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((self.y_depth,), dtype='int32', name='input_y')
        z = zeros_layer(1)(input_x)
        skipgram = SkipgramLayer(k=k, units=units)
        nll = skipgram([z, input_y])

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def avg_nll(ytrue, ypred):
            return T.mean(nll, axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[nll])
        self.model.compile(opt, loss_f, metrics=[avg_nll])

        self.model_encode = Model(inputs=[input_x], outputs=[z])

        srng = RandomStreams(123)
        policy = SkipgramPolicyLayer(skipgram, srng=srng, depth=self.y_depth)
        ypred = policy(z)
        self.model_predict = Model(inputs=[input_x], outputs=[ypred])

    def decode_sample(self, y):
        word = "_"
        ctx = [self.dataset.get_word(y[i]) for i in range(y.shape[0])]
        lctx = ctx[:self.window]
        rctx = ctx[self.window:]
        return "{} [{}] {}".format(" ".join(lctx), word, " ".join(rctx))

    def write_predictions(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        n = 128
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Idx", "Sample"])
            x = np.zeros((n, 1), dtype=np.float32)
            y = self.model_predict.predict(x, verbose=0)
            for i in range(n):
                sample = self.decode_sample(y[i, :])
                w.writerow([i, sample])

    def on_epoch_end(self, output_path, frequency, epoch, logs):
        if (epoch + 1) % frequency == 0:
            self.write_predictions("{}/predictions-{:08d}.csv".format(output_path, epoch))
            self.model.save_weights("{}/model-{:08d}.csv".format(output_path, epoch))

    def train(self, batch_size, epochs, steps_per_epoch, output_path, frequency=10, **kwargs):
        def on_epoch_end(epoch, logs):
            self.on_epoch_end(output_path, frequency, epoch, logs)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)
        csvcb = CSVLogger("{}/history.csv".format(output_path))
        cb = LambdaCallback(on_epoch_end=on_epoch_end)
        self.model.fit_generator(gen, epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=[cb, csvcb],
                                 **kwargs)
