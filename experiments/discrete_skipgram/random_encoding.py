import numpy as np

from discrete_skip_gram.skipgram.corpus import load_corpus

if __name__ == "__main__":
    encoding_path = "output/random_encoding.npy"
    corpus_path = "output/corpus"
    vocab, corpus = load_corpus(corpus_path)
    x_k = len(vocab) + 1
    z_depth = 10
    z_k = 2
    encoding = np.random.randint(0, z_k, (x_k, z_depth))
    np.save(encoding_path, encoding)
    print "Min {}, Max {}".format(np.min(encoding), np.max(encoding))
