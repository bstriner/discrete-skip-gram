import csv
import itertools
import os

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.corpus import load_corpus
from discrete_skip_gram.skipgram.tensor_util import softmax_nd
from keras.optimizers import Adam
from keras.regularizers import L1L2
float_n = np.float32
float_t = 'float32'


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
def calc_depth(depth, p_z, co_pt, z_k, py_weight):
    # p_z: (n, z_depth, z_k)
    # co_pt: (n, x_k)
    # py_weight: (buckets, x_k)
    # Create mask
    buckets = int(2 ** depth)
    print "bits {}, buckets: {}".format(depth, buckets)
    _mask = np.zeros(shape=(depth, z_k, buckets), dtype=float_n)
    for i, prod in enumerate(itertools.product(list(range(z_k)), repeat=depth)):
        for b in range(depth):
            for z in range(z_k):
                if prod[b] == z:
                    _mask[b, z, i] = 1
    mask = T.constant(_mask)  # (depth, z_k, buckets)
    # Bucket softmax
    py = softmax_nd(py_weight)  # (buckets, x_k)
    # Calculate probability by bucket
    p_z_part = p_z[:, :depth, :]  # (n, depth, z_k)
    h = (mask.dimshuffle(('x', 0, 1, 2))) * (p_z_part.dimshuffle((0, 1, 2, 'x')))  # (n, depth, z_k, buckets)
    h = T.sum(h, axis=2)  # (n, depth, buckets)
    h = T.prod(h, axis=1)  # (n, buckets)
    p_b = h  # / (T.sum(h, axis=1, keepdims=True))  # (n, buckets)
    # Calculate loss by bucket
    eps = 1e-8
    h = (co_pt.dimshuffle((0, 'x', 1))) * -T.log(eps + py.dimshuffle(('x', 0, 1)))  # (n, buckets, x_k)
    loss_by_bucket = T.sum(h, axis=2)  # (n, buckets)
    # Total loss
    h = T.sum(loss_by_bucket * p_b, axis=1)  # (n,)
    loss = T.mean(h, axis=0)
    return loss


def build_model(cooccurrence, z_depth, z_k, loss_schedule, corp, batch_size, opt, regularizer=None):
    scale = 1e-1
    x_k = cooccurrence.shape[0]
    schedule = T.constant(loss_schedule)

    # marginal probability
    n = np.sum(cooccurrence, axis=None)
    # _margin = np.sum(cooccurrence, axis=1) / n  # (x_k,)
    # _csum = np.cumsum(_margin)
    # csum = T.constant(_csum)  # (x_k,)

    # conditional probability
    _cond_p = cooccurrence / np.sum(cooccurrence, axis=1, keepdims=True)
    cond_p = T.constant(_cond_p)

    # parameters
    initial_weight = np.random.uniform(-scale, scale, (x_k, z_depth, z_k)).astype(float_n)
    weight = theano.shared(initial_weight, name="weight")  # (x_k, z_depth, z_k)
    pys = []
    for depth in range(z_depth):
        n = int(2 ** (depth + 1))
        initial_py = np.random.uniform(-scale, scale, (n, x_k)).astype(float_n)  # (buckets, x_k)
        py_weight = theano.shared(initial_py, name='py_{}'.format(depth))  # (buckets, x_k)
        pys.append(py_weight)
    params = [weight] + pys

    # indices
    srng = RandomStreams(123)
    rng = srng.random_integers(size=(batch_size,), low=0, high=corp.shape[0] - 1)  # (n,)
    idx = corp[rng]  # (n,)

    # p_z
    h = weight[idx, :, :]  # (n, z_depth, z_k)
    p_z = softmax_nd(h)  # (n, z_depth, z_k)

    co_pt = cond_p[idx, :]  # (n, x_k)

    nlls = []
    for depth in range(1, z_depth + 1):
        nll = calc_depth(depth, p_z, co_pt, z_k, pys[depth - 1])
        nlls.append(nll)
    nlls = T.stack(nlls, axis=1)  # (z_depth,)
    loss = T.sum(schedule * nlls)
    if regularizer:
        for p in params:
            loss += regularizer(p)
    updates = opt.get_updates(params, {}, loss)
    train = theano.function([], [nlls, loss], updates=updates)

    encs = softmax_nd(weight)
    encodings = theano.function([], encs)
    return train, encodings


def main():
    batch_size = 64
    opt = Adam(1e-3)
    regularizer = L1L2(1e-11, 1e-11)
    outputpath = "output/skipgram_discrete_co"
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(float_n)
    vocab, corpus = load_corpus('output/corpus')
    corp = T.constant(corpus, dtype='int32', name='corpus')
    z_depth = 10
    z_k = 2

    loss_schedule = np.power(1.5, np.arange(z_depth))
    train, encodings = build_model(cooccurrence=cooccurrence,
                                   batch_size=batch_size,
                                   opt=opt,
                                   z_depth=z_depth,
                                   loss_schedule=loss_schedule,
                                   regularizer=regularizer,
                                   corp=corp,
                                   z_k=z_k)

    epochs = 1000
    batches = 1024
    loss = None
    nll = None
    beta = 1e-2
    with open(os.path.join(outputpath, 'history.csv'), 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Epoch', 'Loss'] + ['NLL {}'.format(i) for i in range(z_depth)])
        f.flush()
        for epoch in tqdm(range(epochs), desc="Training"):
            it = tqdm(range(batches), desc="Epoch {}".format(epoch))
            for batch in it:
                _nll, _loss = train()
                if nll is None:
                    nll = _nll
                if loss is None:
                    loss = _loss
                nll = (beta * _nll) + ((1. - beta) * nll)
                loss = (beta * _loss) + ((1. - beta) * loss)
                nllstr = ", ".join("{:.2f}".format(np.asscalar(nll[i])) for i in range(z_depth))
                it.desc = "Epoch {} Loss {:.4f} NLL [{}]".format(epoch, np.asscalar(loss), nllstr)
            w.writerow([epoch, loss] + [nll[i] for i in range(z_depth)])
            f.flush()
            enc = encodings()  # (n, z_depth, x_k)
            np.save(os.path.join(outputpath, 'probabilities-{:08d}.npy'.format(epoch)), enc)
            z = np.argmax(enc, axis=2)  # (n, z_depth)
            np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), z)


if __name__ == "__main__":
    main()
