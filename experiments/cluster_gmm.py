import os

import numpy as np

from discrete_skip_gram.clustering.gmm import cluster_gmm
from discrete_skip_gram.clustering.utils import write_encodings
from discrete_skip_gram.corpus import load_corpus
from discrete_skip_gram.models.util import latest_model


def main():
    output_path = "output/cluster_gmm/encodings"
    corpus_path = "output/corpus"
    vocab, corpus = load_corpus(corpus_path)
    if os.path.exists(output_path):
        raise ValueError("Already exists: {}".format(output_path))
    path = "output/skipgram_baseline_co"
    file, epoch = latest_model(path, "encodings-(\\d+).npy")
    print "Loading epoch {}: {}".format(epoch, file)
    z = np.load(file)
    z_depth = 10
    print "Clustering"
    enc = cluster_gmm(z, z_depth)
    print "Writing"
    write_encodings(enc, vocab, output_path)


if __name__ == "__main__":
    main()
