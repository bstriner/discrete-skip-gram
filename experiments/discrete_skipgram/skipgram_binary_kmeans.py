import numpy as np

from discrete_skip_gram.clustering.kmeans import cluster_km
from discrete_skip_gram.clustering.validate_clusters import validate_clustering
from discrete_skip_gram.models.util import latest_model
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.validation import validate_encoding_flat


def main():
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    op = "output/skipgram_binary_kmeans"
    ip = "output/skipgram_baseline"
    z_k = 2
    for z_units in [512, 256, 128, 64, 32]:
        input_path = "{}/{}".format(ip, z_units)
        output_path = "{}/{}.csv".format(op, z_units)
        encoding_path, epoch = latest_model(input_path, "encodings-(\\d+).npy", fail=True)
        print "Loading epoch {}: {}".format(epoch, encoding_path)
        z = np.load(encoding_path)
        iters = 10
        validate_clustering(output_path=output_path,
                            z=z,
                            z_k=z_k, iters=iters,
                            clustering=cluster_km,
                            validation=validate_encoding_flat(cooccurrence=cooccurrence))


if __name__ == "__main__":
    main()
