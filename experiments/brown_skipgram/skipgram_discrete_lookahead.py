# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import sys

print "sys.getrecursionlimit: {}".format(sys.getrecursionlimit())
sys.setrecursionlimit(10000)  # with z_depth=10 you get maximum recursion depth exceeded
import numpy as np

from discrete_skip_gram.dataset_util import load_dataset
from discrete_skip_gram.layers.utils import leaky_relu
from discrete_skip_gram.skipgram_models.skipgram_discrete_lookahead_model import SkipgramDiscreteLookaheadModel
from keras.optimizers import RMSprop, Adam
from keras.regularizers import L1L2


def main():
    outputpath = "output/skipgram_discrete_lookahead"
    ds = load_dataset()
    batch_size = 256
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 20
    units = 512
    embedding_units = 128
    z_k = 2
    lookahead_depth = 10
    z_depth = 10
    adam = True
    if adam:
        #        opt = Adam(3e-4)
        opt = Adam(1e-3)
    else:
        opt = RMSprop(3e-4)
    mode = 2
    layernorm = False
    batchnorm = False
    hidden_layers = 1
    embeddings_regularizer = L1L2(1e-9, 1e-9)
    scale = 1e-1
    growth = 1.5
    floating = 'float64'
    schedule = np.float64(scale * np.power(growth, np.arange(z_depth)))
    model = SkipgramDiscreteLookaheadModel(dataset=ds,
                                           z_k=z_k,
                                           schedule=schedule,
                                           z_depth=z_depth,
                                           window=window,
                                           floating=floating,
                                           embeddings_regularizer=embeddings_regularizer,
                                           embedding_units=embedding_units,
                                           lookahead_depth=lookahead_depth,
                                           opt=opt,
                                           hidden_layers=hidden_layers,
                                           layernorm=layernorm,
                                           batchnorm=batchnorm,
                                           inner_activation=leaky_relu,
                                           mode=mode,
                                           units=units)
    model.summary()
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                output_path=outputpath,
                frequency=frequency)


if __name__ == "__main__":
    main()
