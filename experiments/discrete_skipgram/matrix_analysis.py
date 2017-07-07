import numpy as np

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence

if __name__ == '__main__':
    x = load_cooccurrence('output/cooccurrence.npy')
    print x.shape
    n = np.sum(x, axis=None)
    print "Datapoints: {}".format(n)

    eps = 1e-9
    a_margin = np.sum(x, axis=1, keepdims=True)
    b_margin = np.sum(x, axis=0, keepdims=True)
    p_a = a_margin / n
    p_b = b_margin / n

    # Unigram
    unigram = np.sum(p_a * -np.log(p_a + eps), axis=0)
    print "Unigram: {}".format(unigram)

    # Skipgram
    p_b_a = x / a_margin
    nll = np.sum(p_b_a * -np.log(p_b_a + eps), axis=1, keepdims=True)
    skipgram = np.sum(nll * p_a, axis=0)
    print "Skipgram: {}".format(skipgram)
