# os.environ["THEANO_FLAGS"]='device=cpu,optimizer=None'

import pickle

import numpy as np
from keras.optimizers import Adam
from theano.tensor.shared_randomstreams import RandomStreams
from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.lm.lstm_softmax_vanilla import LSTMSoftmaxVanilla
from discrete_skip_gram.lm.speedtest import run_speedtest

def main():
    output_path = 'output/lstm_softmax_vanilla_speedtest.npz'
    iters = 1000
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
    x_k = len(vocab)
    xtrain, xvalid, xtest = x["train"], x["valid"], x["test"]

    i1 = np.arange(batch_size).reshape((-1,1))
    i2 = np.arange(depth).reshape((1, -1))
    idx = i1+i2
    xb = xtrain[idx]
    model = LSTMSoftmaxVanilla(units=units,
                               x_k=x_k,
                               opt=opt,
                               zoneout=zoneout,
                               srng=srng,
                               input_droput=input_droput,
                               output_dropout=output_dropout,
                               initializer=initializer)

    def train_fun():
        model.train_fun(xb)

    def test_fun():
        model.nll_fun(xb)
    run_speedtest(output_path=output_path, train=train_fun, test=test_fun, iters=iters)


if __name__ == '__main__':
    main()
