import csv
import os

import h5py
import keras.backend as K
import numpy as np
from keras.callbacks import LambdaCallback, CSVLogger

from .util import latest_model


class SkipgramModel(object):
    def save(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with h5py.File(output_path, 'w') as f:
            for i, w in enumerate(self.weights):
                f.create_dataset(name="param_{}".format(i), data=K.get_value(w))

    def load(self, input_path):
        with h5py.File(input_path, 'r') as f:
            for i, w in enumerate(self.weights):
                K.set_value(w, f["param_{}".format(i)])

    def summary(self):
        self.model.summary()

    def continue_training(self, output_path):
        initial_epoch = 0
        ret = latest_model(output_path, "model-(\\d+).h5")
        if ret:
            self.load(ret[0])
            initial_epoch = ret[1] + 1
            print "Resuming training at {}".format(initial_epoch)
        return initial_epoch

    def train(self, x, batch_size, epochs, output_path, frequency=10, continue_training=True, **kwargs):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        initial_epoch = 0
        if continue_training:
            initial_epoch = self.continue_training(output_path)

        def on_epoch_end(epoch, logs):
            if (epoch + 1) % frequency == 0:
                self.on_epoch_end(output_path, epoch)

        y = np.zeros((x.shape[0], 1), dtype='float32')
        csvcb = CSVLogger("{}/history.csv".format(output_path), append=continue_training)
        cb = LambdaCallback(on_epoch_end=on_epoch_end)
        self.model.fit(x, y, epochs=epochs,
                       batch_size=batch_size,
                                 callbacks=[cb, csvcb],
                                 initial_epoch=initial_epoch,
                                 **kwargs)

    def decode_sample(self, x, y):
        word = self.dataset.get_word(x)
        ctx = [self.dataset.get_word(self.hsm.decode(y[i, :])) for i in range(y.shape[0])]
        lctx = ctx[:self.window]
        rctx = ctx[self.window:]
        return "{} [{}] {}".format(" ".join(lctx), word, " ".join(rctx))

    def decode_sample_flat(self, x, y):
        word = self.dataset.get_word(x)
        ctx = [self.dataset.get_word(y[i]) for i in range(y.shape[0])]
        lctx = ctx[:self.window]
        rctx = ctx[self.window:]
        return "{} [{}] {}".format(" ".join(lctx), word, " ".join(rctx))

    def write_predictions(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        samples = 16
        n = 128
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word"] + ["Sample {}".format(i) for i in range(samples)])
            x = np.random.randint(0, self.dataset.k, size=(n, 1))
            ys = [self.model_predict.predict(x, verbose=0) for _ in range(samples)]
            for i in range(n):
                ix = x[i, 0]
                word = self.dataset.get_word(ix)
                samples = [self.dataset.get_word(y[i,0]) for y in ys]
                w.writerow([ix, word] + samples)

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