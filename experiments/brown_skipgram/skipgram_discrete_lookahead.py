#import os
#os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np

from dataset_util import load_dataset
from discrete_skip_gram.layers.utils import leaky_relu
from discrete_skip_gram.skipgram_models.skipgram_discrete_lookahead_model import SkipgramDiscreteLookaheadModel
from keras.optimizers import Adam


def main():
    outputpath = "output/skipgram_discrete_lookahead"
    ds = load_dataset()
    batch_size = 256
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 10
    units = 512
    embedding_units = 128
    z_k = 2
    z_depth = 10
    opt = Adam(3e-4)
    layernorm = False
    batchnorm = False
    hidden_layers = 1
    schedule = np.float32(np.power(1.2, np.arange(z_depth)))
    model = SkipgramDiscreteLookaheadModel(dataset=ds,
                                           z_k=z_k,
                                           schedule=schedule,
                                           z_depth=z_depth,
                                           window=window,
                                           embedding_units=embedding_units,
                                           opt=opt,
                                           hidden_layers=hidden_layers,
                                           layernorm=layernorm,
                                           batchnorm=batchnorm,
                                           inner_activation=leaky_relu,
                                           units=units)
    model.summary()
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                output_path=outputpath,
                frequency=frequency)


if __name__ == "__main__":
    main()
