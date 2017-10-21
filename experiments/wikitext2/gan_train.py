# import os
# os.environ["THEANO_FLAGS"]='device=cpu,optimizer=None'

import pickle

import numpy as np
from keras.optimizers import Adam
from keras.regularizers import l2
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_skip_gram.constraints import clip_constraint
from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.lm.ganlm import GANLanguageModel


def main():
    output_path = 'output/ganlm'
    epochs = 100
    batches = 1000
    val_batches = 1000
    d_batches = 5
    depth = 12
    batch_size = 64
    tau0 = 5.
    tau_decay = 1e-5
    tau_min = 0.2
    d_layers = 2
    g_layers = 2
    d_units = 512
    g_units = 512
    dopt = Adam(1e-3)
    gopt = Adam(1e-4)
    regularizer = l2(1e-5)
    regularizer_samples = 128
    regularizer_weight = 1e1
    #constraint = clip_constraint(1e-1)
    constraint = None
    initializer = uniform_initializer(0.05)
    srng = RandomStreams(123)
    x = np.load('output/corpus/corpus.npz')
    with open('output/corpus/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    xtrain, xvalid, xtest = x["train"], x["valid"], x["test"]

    model = GANLanguageModel(vocab=vocab,
                             d_units=d_units,
                             g_units=g_units,
                             initializer=initializer,
                             dopt=dopt,
                             gopt=gopt,
                             srng=srng,
                             d_layers=d_layers,
                             g_layers=g_layers,
                             regularizer=regularizer,
                             regularizer_samples=regularizer_samples,
                             regularizer_weight=regularizer_weight,
                             constraint=constraint,
                             tau0=tau0,
                             tau_decay=tau_decay,
                             tau_min=tau_min)
    model.train(output_path=output_path,
                epochs=epochs,
                batches=batches,
                depth=depth,
                d_batches=d_batches,
                batch_size=batch_size,
                val_batches=val_batches,
                xtrain=xtrain,
                xvalid=xvalid,
                xtest=xtest)


if __name__ == '__main__':
    main()
