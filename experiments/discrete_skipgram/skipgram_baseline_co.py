# os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"

# os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"
import csv
import os

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from keras.optimizers import Adam
from keras.regularizers import L1L2

float_n = np.float32
float_t = 'float32'


def build_model(cooccurrence, z_units, opt, regularizer=None):
    scale = 1e-1
    x_k = cooccurrence.shape[0]

    # parameters
    initial_embedding = np.random.uniform(-scale, scale, (x_k, z_units)).astype(float_n)
    initial_weight = np.random.uniform(-scale, scale, (z_units, x_k)).astype(float_n)
    initial_bias = np.random.uniform(-scale, scale, (x_k,)).astype(float_n)
    embedding = theano.shared(initial_embedding, name="embedding")
    weight = theano.shared(initial_weight, name="weight")
    bias = theano.shared(initial_bias, name="bias")

    # conditional probability
    _cond_p = cooccurrence / np.sum(cooccurrence, axis=1, keepdims=True)
    cond_p = T.constant(_cond_p)

    # marginal probability
    n = np.sum(cooccurrence, axis=None)
    _marg_p = np.sum(cooccurrence, axis=1) / n
    marg_p = T.constant(_marg_p)

    # p
    h = T.dot(embedding, weight) + bias
    p = T.nnet.softmax(h)  # (x_k, x_k)
    nll_part = T.sum(cond_p * -T.log(p), axis=1)  # (x_k,)
    nll = T.sum(marg_p * nll_part, axis=0)  # scalar
    loss = nll
    if regularizer:
        reg = regularizer(weight) + regularizer(embedding)
        loss += reg

    params = [embedding, weight, bias]
    updates = opt.get_updates(params, {}, loss)
    train = theano.function([], [nll, loss], updates=updates)

    encodings = theano.function([], embedding)
    return train, encodings


def main():
    opt = Adam(1e-3)
    outputpath = "output/skipgram_baseline_co"
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(float_n)
    z_units = 256
    regularizer = L1L2(1e-8, 1e-8)
    train, encodings = build_model(cooccurrence=cooccurrence,
                                   regularizer=regularizer,
                                   opt=opt,
                                   z_units=z_units)

    epochs = 1000
    batches = 1024
    with open(os.path.join(outputpath, 'history.csv'), 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Epoch', 'NLL', 'Loss'])
        f.flush()
        beta = 1e-2
        loss = None
        nll = None
        for epoch in tqdm(range(epochs), desc="Training"):
            it = tqdm(range(batches), desc="Epoch {}".format(epoch))
            for batch in it:
                _nll, _loss = train()
                if not loss:
                    loss = _loss
                if not nll:
                    nll = _nll
                loss = (beta * _loss) + ((1. - beta) * loss)
                nll = (beta * _nll) + ((1. - beta) * nll)
                it.desc = "Epoch {} NLL {:.04f} Loss {:.04f}".format(epoch, np.asscalar(nll), np.asscalar(loss))
            w.writerow([epoch, np.asscalar(nll), np.asscalar(loss)])
            f.flush()
            z = encodings()  # (n, z_units)
            np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), z)


if __name__ == "__main__":
    main()
