import numpy as np

from discrete_skip_gram.clustering.kmeans import cluster_km
from discrete_skip_gram.clustering.validate_clusters import validate_clusters
from discrete_skip_gram.models.util import latest_model
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.validation import validate_encoding_flat


def main():
    output_path = "output/skipgram_flat_kmeans.csv"
    inputpath = "output/skipgram_baseline"

    file, epoch = latest_model(inputpath, "encodings-(\\d+).npy", fail=True)
    print "Loading epoch {}: {}".format(epoch, file)
    z = np.load(file)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    zks = [(2 ** i) for i in range(1, 11)]
    iters = 2
    validate_clusters(output_path=output_path, z=z, cooccurrence=cooccurrence,
                      zks=zks, iters=iters, clustering=cluster_km,
                      validation=validate_encoding_flat(cooccurrence=cooccurrence))


if __name__ == "__main__":
    main()
