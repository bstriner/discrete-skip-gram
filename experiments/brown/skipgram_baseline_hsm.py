# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

import numpy as np
from keras.regularizers import L1L2

from dataset import load_dataset
from discrete_skip_gram.models.word_skipgram_baseline_hsm import WordSkipgramBaselineHSM
from random_hsm import load_hsm
from sample_validation import validation_load


def main():
    outputpath = "output/brown/skipgram_baseline_hsm"
    dataset = load_dataset()
    hsm = load_hsm()
    vd = validation_load()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 512
    frequency = 25
    kernel_regularizer = None #L1L2(1e-4, 1e-4)
    window = 2
    units = 256
    embedding_units = 128
    lr = 1e-3

    model = WordSkipgramBaselineHSM(dataset=dataset,
                                    embedding_units=embedding_units,
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
