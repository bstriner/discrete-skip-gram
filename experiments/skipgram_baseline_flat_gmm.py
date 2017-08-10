import numpy as np
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence

from discrete_skip_gram.clustering.gmm import cluster_gmm_flat
from discrete_skip_gram.clustering.validate_clusters import validate_clusters
from discrete_skip_gram.models.util import latest_model
from discrete_skip_gram.flat_validation import validate_encoding_flat


def main():
    output_path = "output/skipgram_baseline_flat_gmm"
    inputpath = "output/skipgram_baseline"
    baseline_z_ks = [512, 256, 128, 64, 32]
    baseline_iters = 5
    z_k = 1024

    for baseline_z_k in baseline_z_ks:



    file, epoch = latest_model(inputpath, "encodings-(\\d+).npy", fail=True)
    print "Loading epoch {}: {}".format(epoch, file)
    z = np.load(file)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    zks = [(2 ** i) for i in range(1, 11)]
    iters = 5
    validate_clusters(output_path=output_path, z=z, cooccurrence=cooccurrence,
                      zks=zks, iters=iters, clustering=cluster_gmm_flat,
                      validation=validate_encoding_flat(cooccurrence=cooccurrence))


if __name__ == "__main__":
    main()
