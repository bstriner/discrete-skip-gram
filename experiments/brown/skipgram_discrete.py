# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import csv
import os

import numpy as np
from keras.callbacks import CSVLogger

from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.datasets.utils import clean_docs, simple_clean
from discrete_skip_gram.datasets.word_dataset import WordDataset
from discrete_skip_gram.models.util import makepath
from discrete_skip_gram.models.word_skipgram_discrete import WordSkipgramDiscrete
from discrete_skip_gram.regularizers.tanh_regularizer import TanhRegularizer
from dataset import load_dataset
from keras.regularizers import L1L2
# maybe try movie_reviews or reuters
def main():
    outputpath = "output/brown/skipgram_discrete"
    ds = load_dataset()

    batch_size = 128
    epochs = 5000
    steps_per_epoch = 512
    window = 2
    frequency = 25
    units = 128
    z_k = 2
    z_depth = 10
    kernel_regularizer = L1L2(1e-7, 1e-7)
    lr = 1e-3
    adversary_weight = 1.0
    model = WordSkipgramDiscrete(dataset=ds, z_k=z_k, z_depth=z_depth,
                                       window=window,
                                 kernel_regularizer=kernel_regularizer,
                                 adversary_weight=adversary_weight,
                                       units=units, lr=lr)
    csvpath = "{}/history.csv".format(outputpath)
    makepath(csvpath)
    if os.path.exists(csvpath):
        os.remove(csvpath)
    validation_n = 4096
    vd = ds.cbow_batch(n=validation_n, window=window, test=True)

    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=([vd[1], vd[0]], np.ones((validation_n, 1), dtype=np.float32)),
                output_path=outputpath,
                frequency=frequency)


if __name__ == "__main__":
    main()
