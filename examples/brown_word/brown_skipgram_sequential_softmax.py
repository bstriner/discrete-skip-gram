# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import os

import numpy as np
from keras.regularizers import L1L2

from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.datasets.utils import clean_docs, simple_clean
from discrete_skip_gram.datasets.word_dataset import WordDataset
from discrete_skip_gram.models.util import makepath
from discrete_skip_gram.models.word_skipgram_sequential_softmax import WordSkipgramSequentialSoftmax
from discrete_skip_gram.regularizers.tanh_regularizer import TanhRegularizer


# maybe try movie_reviews or reuters
def main():
    outputpath = "output/brown/skipgram_sequential_softmax"
    min_count = 5
    batch_size = 128
    epochs = 500
    # epochs = 5
    steps_per_epoch = 256
    # steps_per_epoch = 2
    window = 2
    units = 128
    z_k = 4
    z_depth = 7
    # 4^6 = 4096
    decay = 0.9
    schedule = np.power(decay, np.arange(z_depth))
    # reg = L1L2(1e-6, 1e-6)
    #reg = None
    #act_reg = TanhRegularizer(1e-3)
    #balance_reg = 1e-2
    #certainty_reg = 1e-2
    #kernel_regularizer = L1L2(1e-6, 1e-6)
    kernel_regularizer = None
    # balance_reg = 0
    # certainty_reg = 0
    lr = 3e-4
    lr_a = 3e-4
    docs = clean_docs(brown_docs(), simple_clean)
    docs, tdocs = docs[:-5], docs[-5:]
    ds = WordDataset(docs, min_count, tdocs=tdocs)
    ds.summary()
    adversary_weight = 1.0
    model = WordSkipgramSequentialSoftmax(dataset=ds, z_k=z_k, z_depth=z_depth,
                                           window=window,
                                           kernel_regularizer=kernel_regularizer,
                                           schedule=schedule,
                                          adversary_weight=adversary_weight,
                                          lr_a=lr_a,
                                           # reg=reg, act_reg=act_reg,
                                           # balance_reg=balance_reg,
                                           # certainty_reg=certainty_reg,
                                           units=units, lr=lr)
    validation_n = 4096
    vd = ds.cbow_batch(n=validation_n, window=window, test=True)

    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=([vd[1], vd[0]], np.ones((validation_n, 1), dtype=np.float32)),
                output_path=outputpath)


if __name__ == "__main__":
    main()
