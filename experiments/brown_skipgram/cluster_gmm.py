import os

import numpy as np

from discrete_skip_gram.clustering.gmm import cluster_gmm
from discrete_skip_gram.clustering.utils import write_encodings
from discrete_skip_gram.dataset_util import load_dataset
from discrete_skip_gram.models.util import latest_model


def main():
    output_path = "output/cluster_gmm/encodings"
    if os.path.exists(output_path):
        raise ValueError("Already exists: {}".format(output_path))
    path = "output/skipgram_baseline"
    file, epoch = latest_model(path, "encodings-(\\d+).npy")
    print "Loading epoch {}: {}".format(epoch, file)
    z = np.load(file)
    z_depth = 10
    enc = cluster_gmm(z, z_depth)
    ds = load_dataset()
    write_encodings(enc, ds, output_path)


if __name__ == "__main__":
    main()
