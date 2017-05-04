import h5py
import keras.backend as K
import os
import os

import h5py
import keras.backend as K


class SGModel(object):

    def save(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with h5py.File(output_path, 'w') as f:
            for i,w in enumerate(self.weights):
                f.create_dataset("param_{}".format(i), K.get_value(w), dtype='float32')

    def load(self, input_path):
        with h5py.File(input_path, 'r') as f:
            for i,w in enumerate(self.weights):
                K.set_value(w, f["param_{}".format(i)])

    