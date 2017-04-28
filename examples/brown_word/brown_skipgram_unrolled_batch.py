# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import csv
import os

import numpy as np
from keras.callbacks import CSVLogger

from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.datasets.utils import clean_docs, simple_clean
from discrete_skip_gram.datasets.word_dataset import WordDataset
from discrete_skip_gram.models.util import makepath
from discrete_skip_gram.models.word_skipgram_unrolled_batch import WordSkipgramUnrolledBatch
from discrete_skip_gram.regularizers.tanh_regularizer import TanhRegularizer


# maybe try movie_reviews or reuters
def main():
    outputpath = "output/brown/skipgram_unrolled_batch"
    min_count = 5
    batch_size = 128
    epochs = 1000
    #epochs = 10
    steps_per_epoch = 256
    #steps_per_epoch = 2
    window = 2
    units = 128
    z_k = 4
    z_depth = 1
    # 4^6 = 4096
    decay = 0.9
    # reg = L1L2(1e-6, 1e-6)
    reg = None
    #act_reg = TanhRegularizer(1e-3)
    act_reg = None
    #balance_reg = 1e-2
    #certainty_reg = 1e-2
    balance_reg = 0
    certainty_reg = 0
    lr = 3e-4

    docs = clean_docs(brown_docs(), simple_clean)
    docs, tdocs = docs[:-5], docs[-5:]
    ds = WordDataset(docs, min_count, tdocs=tdocs)
    ds.summary()
    x_k = ds.k
    model = WordSkipgramUnrolledBatch(dataset=ds, z_k=z_k, z_depth=z_depth,
                                       window=window,
                                       reg=reg, act_reg=act_reg,
                                       balance_reg=balance_reg,
                                       certainty_reg=certainty_reg,
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
                output_path=outputpath)

if __name__ == "__main__":
    main()
