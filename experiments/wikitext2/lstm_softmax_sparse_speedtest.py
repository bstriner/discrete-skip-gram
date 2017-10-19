# os.environ["THEANO_FLAGS"]='device=cpu,optimizer=None'

import pickle

import numpy as np
from keras.optimizers import Adam
from keras.regularizers import l2
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.lm.lstm_softmax_sparse import LSTMSoftmaxSparse
from discrete_skip_gram.lm.speedtest import run_speedtest


def sparse_speedtest(output_path, encoding):
    iters = 1000
    units = 512
    depth = 35
    batch_size = 64
    input_droput = 0.
    zoneout = 0.5
    output_dropout = 0.
    activity_reg = 0.
    temporal_activity_reg = 0.
    layers = 1
    initializer = uniform_initializer(0.05)
    opt = Adam(1e-3)
    srng = RandomStreams(123)
    x = np.load('output/corpus/corpus.npz')
    with open('output/corpus/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    xtrain, xvalid, xtest = x["train"], x["valid"], x["test"]

    i1 = np.arange(batch_size).reshape((-1, 1))
    i2 = np.arange(depth).reshape((1, -1))
    idx = i1 + i2
    xb = xtrain[idx]
    model = LSTMSoftmaxSparse(units=units,
                              encoding=encoding,
                              vocab=vocab,
                              opt=opt,
                              zoneout=zoneout,
                              srng=srng,
                              layers=layers,
                              activity_reg=activity_reg,
                              temporal_activity_reg=temporal_activity_reg,
                              input_droput=input_droput,
                              output_dropout=output_dropout,
                              initializer=initializer)

    def train_fun():
        model.train_fun(xb)

    def test_fun():
        model.nll_fun(xb)

    return run_speedtest(output_path=output_path, train=train_fun, test=test_fun, iters=iters)


def main():
    output_path = 'output/lstm_softmax_sparse_speedtest'
    units = [16, 32, 64, 128, 256]
    input_path = 'output/random_encodings'

    trains = []
    tests = []
    for u in tqdm(units, desc='Speed Test'):
        encoding = np.load('{}/units-{}/iter-0.npy'.format(input_path, u))
        train, test = sparse_speedtest(output_path='{}/units-{}.npz'.format(output_path, u),
                                       encoding=encoding)
        trains.append(train)
        tests.append(test)
    np.savez(output_path + 'npz',
             units=np.array(units),
             train_times=np.array(trains),
             test_times=np.array(tests))

    for i, u in enumerate(units):
        print("Units: {}, Train: {}ms, Test {}ms".format(u, trains[i] * 1000, tests[i] * 1000))


if __name__ == '__main__':
    main()
