import csv
import os

import h5py
import keras.backend as K
import numpy as np
from keras.callbacks import LambdaCallback, CSVLogger

from ..datasets.mnist_dataset import mnist_generator, mnist_val, mnist_data
from ..skipgram_models.util import latest_model


class MnistModel(object):
    def __init__(self):
        self.train_data, self.test_data = mnist_data()

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

    def train(self, batch_size, epochs, steps_per_epoch, output_path, frequency=10, continue_training=True, **kwargs):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        initial_epoch = 0
        if continue_training:
            initial_epoch = self.continue_training(output_path)

        def on_epoch_end(epoch, logs):
            if (epoch + 1) % frequency == 0:
                self.on_epoch_end(output_path, epoch)

        gen = mnist_generator(batch_size)
        csvcb = CSVLogger("{}/history.csv".format(output_path), append=continue_training)
        cb = LambdaCallback(on_epoch_end=on_epoch_end)
        self.model.fit_generator(gen, epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=[cb, csvcb],
                                 initial_epoch=initial_epoch,
                                 validation_data=mnist_val(),
                                 **kwargs)

    def write_encodings(self, output_path):
        z = self.model_encode.predict(self.test_data[0], verbose=0)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path + ".csv", 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Label"] + ["Z {}".format(i) for i in range(self.z_k)])
            for label in range(10):
                where = np.nonzero(np.equal(self.test_data[1], label))
                zwhere = z[where[0], :]
                vals = [np.count_nonzero(np.equal(zwhere, cat)) for cat in range(self.z_k)]
                w.writerow([label] + vals)
