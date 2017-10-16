# os.environ["THEANO_FLAGS"]='device=cpu,optimizer=None'

import pickle

import numpy as np
from keras.optimizers import Adam
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.lm.lstm_softmax_vanilla import LSTMSoftmaxVanilla


def main():
    output_path = 'output/lstm_softmax_vanilla'
    epochs = 100
    batches = 10000
    units = 1024
    depth = 35
    batch_size = 64
    zoneout = 0.5
    input_droput = 0.5
    output_dropout = 0.5
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
