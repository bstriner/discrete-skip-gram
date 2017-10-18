# os.environ["THEANO_FLAGS"]='device=cpu,optimizer=None'

import pickle

import numpy as np
from keras.optimizers import Adam
from keras.regularizers import l2
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.lm.lstm_softmax_vanilla import LSTMSoftmaxVanilla


def main():
    output_path = 'output/lstm_softmax_vanilla'
    epochs = 100
    batches = 5000
    units = 512
    depth = 35
    batch_size = 64
    input_droput = 0.1
    zoneout = 0.5
    output_dropout = 0.5
    regularizer = l2(1e-6)
    activity_reg = 2e-1
    temporal_activity_reg = 1e-1
    layers = 2
    initializer = uniform_initializer(0.05)
    opt = Adam(1e-3)
    srng = RandomStreams(123)
    x = np.load('output/corpus/corpus.npz')
    with open('output/corpus/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    xtrain, xvalid, xtest = x["train"], x["valid"], x["test"]

    model = LSTMSoftmaxVanilla(units=units,
                               vocab=vocab,
                               opt=opt,
                               zoneout=zoneout,
                               srng=srng,
                               layers=layers,
                               regularizer=regularizer,
                               activity_reg=activity_reg,
                               temporal_activity_reg=temporal_activity_reg,
                               input_droput=input_droput,
                               output_dropout=output_dropout,
                               initializer=initializer)
    model.train(output_path=output_path,
                epochs=epochs,
                batches=batches,
                depth=depth,
                batch_size=batch_size,
                xtrain=xtrain,
                xvalid=xvalid,
                xtest=xtest)


if __name__ == '__main__':
    main()
