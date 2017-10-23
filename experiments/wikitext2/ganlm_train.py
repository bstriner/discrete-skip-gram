#import os
#os.environ["THEANO_FLAGS"]='device=cpu,optimizer=None'

import pickle

import numpy as np
from keras.optimizers import Adam
from keras.regularizers import l2
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.lm.gan_language_model import GANModel


def main():
    output_path = 'output/ganlm2'
    epochs = 100
    batches = 1000
    val_batches = 1000
    depth = 20
    batch_size = 64
    d_batches = 5

    dopt = Adam(1e-3)
    d_layers = 2
    d_units = 512
    d_input_dropout = 0.1
    d_zoneout = 0.5
    d_dropout = 0.5
    d_regularizer = l2(1e-4)

    gopt = Adam(1e-5)
    g_layers = 2
    g_units = 1024
    g_input_dropout = 0.1
    g_zoneout = 0.5
    g_dropout = 0.5
    g_regularizer = l2(1e-6)

    regularizer_weight = 1e2
    # constraint = clip_constraint(1e-1)
    constraint = None
    initializer = uniform_initializer(0.05)
    srng = RandomStreams(123)
    x = np.load('output/corpus/corpus.npz')
    with open('output/corpus/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    xtrain, xvalid, xtest = x["train"], x["valid"], x["test"]

    model = GANModel(vocab=vocab,
                     initializer=initializer,
                     dopt=dopt,
                     gopt=gopt,
                     srng=srng,
                     d_layers=d_layers,
                     d_units=d_units,
                     d_input_dropout=d_input_dropout,
                     d_dropout=d_dropout,
                     d_zoneout=d_zoneout,
                     g_layers=g_layers,
                     g_units=g_units,
                     g_input_dropout=g_input_dropout,
                     g_zoneout=g_zoneout,
                     g_dropout=g_dropout,
                     d_regularizer=d_regularizer,
                     g_regularizer=g_regularizer,
                     regularizer_weight=regularizer_weight,
                     constraint=constraint)
    model.train(output_path=output_path,
                epochs=epochs,
                batches=batches,
                depth=depth,
                batch_size=batch_size,
                val_batches=val_batches,
                xtrain=xtrain,
                xvalid=xvalid,
                d_batches=d_batches,
                xtest=xtest)


if __name__ == '__main__':
    main()
