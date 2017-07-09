import csv
import itertools
import os

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.tensor_util import softmax_nd
from keras.optimizers import Adam
from keras.regularizers import L1L2

float_n = np.float32
float_t = 'float32'


def scan_fun(co, nll):
    # co: (x_k,) conditional p of y
    # nll: (buckets, x_k) -log(p(y))
    h = (co.dimshuffle(('x',0)))*nll #(buckets, x_k)
    loss = T.sum(h, axis=1) #(buckets,)
    return loss

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
    eps = 1e-8
    nlly = -T.log(eps+py)
    # Calculate probability by bucket
    p_z_part = p_z[:, :depth, :]  # (x_k, depth, z_k)
    h = (mask.dimshuffle(('x', 0, 1, 2))) * (p_z_part.dimshuffle((0, 1, 2, 'x')))  # (x_k, depth, z_k, buckets)
    h = T.sum(h, axis=2)  # (x_k, depth, buckets)
    h = T.prod(h, axis=1)  # (x_k, buckets)
    p_b = h  # / (T.sum(h, axis=1, keepdims=True))  # (x_k, buckets)
    # Calculate loss by bucket
    loss_by_bucket, _ = theano.scan(scan_fun, sequences=[co_pt], non_sequences=[nlly]) # (x_k, buckets)
    #h = (co_pt.dimshuffle((0, 'x', 1))) * -T.log(eps + py.dimshuffle(('x', 0, 1)))  # (x_k, buckets, x_k)
    #loss_by_bucket = T.sum(h, axis=2)  # (x_k, buckets)
    # Total loss
    loss = T.sum(loss_by_bucket * p_b, axis=1)  # (n,)
    return loss


def build_model(cooccurrence, z_depth, z_k, loss_schedule, opt, regularizer=None):
    scale = 1e-1
    x_k = cooccurrence.shape[0]
    schedule = T.constant(loss_schedule)

    # marginal probability
    n = np.sum(cooccurrence, axis=None)
    _margin = np.sum(cooccurrence, axis=1) / n  # (x_k,)
    marg_p = T.constant(_margin)
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
        bucket_n = int(2 ** (depth + 1))
        initial_py = np.random.uniform(-scale, scale, (bucket_n, x_k)).astype(float_n)  # (buckets, x_k)
        py_weight = theano.shared(initial_py, name='py_{}'.format(depth))  # (buckets, x_k)
        pys.append(py_weight)
    params = [weight] + pys

    # p_z
    p_z = softmax_nd(weight)  # (n, z_depth, z_k)

    nlls = []
    for depth in range(1, z_depth + 1):
        nll = calc_depth(depth, p_z, cond_p, z_k, pys[depth - 1])
        nlls.append(nll)
    nlls = T.stack(nlls, axis=1)  # (n, z_depth)
    wnlls = T.sum((marg_p.dimshuffle((0, 'x'))) * nlls, axis=0)  # (z_depth,)
    loss = T.sum(schedule * wnlls)  # scalar
    if regularizer:
        for p in params:
            loss += regularizer(p)
    updates = opt.get_updates(params, {}, loss)
    train = theano.function([], [wnlls, loss], updates=updates)

    encs = softmax_nd(weight)
    encodings = theano.function([], encs)
    return train, encodings


def train_batch(idx, train, z_depth, batch_size=32):
    return train()


def main():
    batch_size = 64
    opt = Adam(1e-3)
    regularizer = L1L2(1e-12, 1e-12)
    outputpath = "output/skipgram_discrete_co"
    z_depth = 10
    z_k = 2
    epochs = 1000
    batches = 64
    reward_scale = 1.5

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(float_n)

    loss_schedule = np.power(reward_scale, np.arange(z_depth))
    train, encodings = build_model(cooccurrence=cooccurrence,
                                   opt=opt,
                                   z_depth=z_depth,
                                   loss_schedule=loss_schedule,
                                   regularizer=regularizer,
                                   z_k=z_k)

    with open(os.path.join(outputpath, 'history.csv'), 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Epoch', 'Loss'] + ['NLL {}'.format(i) for i in range(z_depth)])
        f.flush()
        idx = np.arange(cooccurrence.shape[0]).astype(np.int32)
        for epoch in tqdm(range(epochs), desc="Training"):
            it = tqdm(range(batches), desc="Epoch {}".format(epoch))
            for batch in it:
                nll, loss = train_batch(idx=idx, train=train, z_depth=z_depth, batch_size=batch_size)
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
