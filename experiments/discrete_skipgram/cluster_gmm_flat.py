import os

import numpy as np

from discrete_skip_gram.clustering.gmm import cluster_gmm_flat
from discrete_skip_gram.clustering.utils import write_encodings
from discrete_skip_gram.dataset_util import load_dataset
from discrete_skip_gram.models.util import latest_model


from discrete_skip_gram.skipgram.corpus import load_corpus

def main():
    output_path = "output/cluster_gmm_flat/encodings"
    corpus_path = "output/corpus"
    vocab, corpus = load_corpus(corpus_path)
    if os.path.exists(output_path):
        raise ValueError("Already exists: {}".format(output_path))
    path = "output/skipgram_baseline_co"
    file, epoch = latest_model(path, "encodings-(\\d+).npy", fail=True)
    print "Loading epoch {}: {}".format(epoch, file)
    z = np.load(file)
    k = 1024
    print "Clustering"
    enc = cluster_gmm_flat(z, k)
    print "Writing"
    write_encodings(enc, vocab, output_path)


if __name__ == "__main__":
    main()
