# import os

# os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"

import numpy as np
from keras.regularizers import L1L2

from dataset import load_dataset
from discrete_skip_gram.models.word_skipgram_baseline_hsm_relu_flat import WordSkipgramBaselineHSMReluFlat
from random_hsm import load_hsm
from sample_validation import validation_load
import theano.tensor as T


def leaky_relu(x):
    return T.nnet.relu(x, 0.2)


def main():
    outputpath = "output/brown/skipgram_baseline_hsm_relu_flat"
    dataset = load_dataset()
    hsm = load_hsm()
    vd = validation_load()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 512
    frequency = 20
    kernel_regularizer = L1L2(1e-7, 1e-7)
    window = 2
    units = 512
    embedding_units = 128
    lr = 1e-3

    model = WordSkipgramBaselineHSMReluFlat(dataset=dataset,
                                            embedding_units=embedding_units,
                                            inner_activation=leaky_relu,
                                            hsm=hsm,
                                            window=window,
                                            kernel_regularizer=kernel_regularizer,
                                            units=units, lr=lr)
    model.summary()
    vn = 2048
    model.train(batch_size=batch_size,
                epochs=epochs,
                frequency=frequency,
                steps_per_epoch=steps_per_epoch,
                validation_data=([vd[0][:vn], vd[1][:vn]], np.ones((vn, 1), dtype=np.float32)),
                output_path=outputpath)


if __name__ == "__main__":
    main()
