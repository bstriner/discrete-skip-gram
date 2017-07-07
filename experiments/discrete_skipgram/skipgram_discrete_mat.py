# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

import itertools
import os

import numpy as np
import theano
import theano.tensor as T

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.corpus import load_corpus
from discrete_skip_gram.skipgram.tensor_util import softmax_nd
from keras.optimizers import Adam


def calc_depth(depth, p_z, co, z_k):
    # p_z: (x_k, z_depth, z_k)
    # co: (x_k, x_k)
    # Create mask
    buckets = int(2 ** depth)
    print "bits {}, buckets: {}".format(depth, buckets)
    _mask = np.zeros(shape=(depth, z_k, buckets), dtype=np.float32)
    for i, prod in enumerate(itertools.product(list(range(z_k)), repeat=depth)):
        for b in range(depth):
            for z in range(z_k):
                if prod[b] == z:
                    _mask[b, z, i] = 1
    mask = T.constant(_mask)  # (depth, z_k, buckets)
    # Calculate probability by bucket
    p_z_part = p_z[:, :depth, :]  # (x_k, depth, z_k)
    h = (mask.dimshuffle(('x', 0, 1, 2))) * (p_z_part.dimshuffle((0, 1, 2, 'x')))  # (x_k, depth, z_k, buckets)
    h = T.sum(h, axis=2)  # (x_k, depth, buckets)
    p_b = T.prod(h, axis=1)  # (x_k, buckets)
    # p_b should sum to 1, should assert
    h = (co.dimshuffle((0, 1, 'x'))) * (p_b.dimshuffle((0, 'x', 1)))  # (x_k, x_k, buckets)
    bucketsum = T.sum(h, axis=0)  # (x_k, buckets)
    bucketmargin = T.sum(bucketsum, axis=0)  # (buckets,)
    n = T.sum(co, axis=None)
    bucketprob = bucketmargin / n  # (buckets,)
    p = bucketsum / T.sum(bucketsum, axis=0, keepdims=True)  # (x_k, buckets)
    entpart = T.sum(p * -T.log(p), axis=0)  # (buckets,)
    entropy = T.sum(entpart * bucketprob, axis=0)  # scalar
    return entropy


def main():
    outputpath = "output/skipgram_discrete_mat"
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    corpus_path = "output/corpus"
    vocab, corpus = load_corpus(corpus_path)
    x = load_cooccurrence('output/cooccurrence.npy')
    co = T.constant(x)
    z_depth = 3
    z_k = 2
    x_k = len(vocab) + 1
    _schedule = np.power(1.5, np.arange(z_depth))
    schedule = T.constant(_schedule)
    print "x_k: {}".format(x_k)
    print "x.shape: {}".format(x.shape)
    assert x_k == x.shape[0]
    assert x_k == x.shape[1]
    scale = 1e-2
    _weight = np.random.uniform(-scale, scale, (x_k, z_depth, z_k))

    weight = theano.shared(_weight, name="weight")  # (x_k, z_depth, z_k)
    p_z = softmax_nd(weight)
    nlls = []
    for depth in range(1, z_depth + 1):
        nll = calc_depth(depth, p_z, co, z_k)
        nlls.append(nll)
    nlls = T.stack(nlls, axis=0)  # (z_depth,)
    loss = T.sum(schedule * nlls)
    opt = Adam(1e-3)
    updates = opt.get_updates([weight], {}, loss)

    train = theano.function([], [nlls, loss], updates=updates)
    encodings = theano.function([], p_z)

    epochs = 1000
    batches = 512
    for epoch in range(epochs):
        for batch in range(batches):
            _nlls, _loss = train()
            print _nlls
        enc = encodings()  # (n, z_depth, x_k)
        np.save(os.path.join(outputpath, 'probabilities-{:08d}.npy'.format(epoch)), enc)
        z = np.argmax(enc, axis=2)  # (n, z_depth)
        np.save(os.path.join(outputpath, 'encodings-{:08d}.npy'.format(epoch)), z)



if __name__ == "__main__":
    main()
