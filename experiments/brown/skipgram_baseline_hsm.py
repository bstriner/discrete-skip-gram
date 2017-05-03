import os
#os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

import numpy as np

from dataset import load_dataset
from sample_validation import validation_load
from random_hsm import load_hsm
from discrete_skip_gram.models.word_skipgram_baseline_hsm import WordSkipgramBaselineHSM


def main():
    outputpath = "output/brown/skipgram_baseline_hsm"
    dataset = load_dataset()
    hsm = load_hsm()
    vd = validation_load()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 512
    frequency = 50
    window = 2
    units = 128
    lr = 1e-3

    model = WordSkipgramBaselineHSM(dataset=dataset,
                                    hsm=hsm,
                                 window=window,
                                 units=units, lr=lr)
    model.summary()
    model.train(batch_size=batch_size,
                epochs=epochs,
                frequency=frequency,
                steps_per_epoch=steps_per_epoch,
                validation_data=(vd, np.ones((vd[0].shape[0], 1), dtype=np.float32)),
                output_path=outputpath)


if __name__ == "__main__":
    main()
