from discrete_skip_gram.clustering.utils import write_encodings
from discrete_skip_gram.clustering.agglomerative import cluster_agglomerative
from discrete_skip_gram.models.util import latest_model
import numpy as np
from dataset_util import load_dataset
import os


def main():
    path = "output/skipgram_baseline"
    output_path = "output/cluster_agglomerative/encodings"
    if os.path.exists(output_path):
        raise ValueError("Already exists: {}".format(output_path))

    file, epoch = latest_model(path, "encodings-(\\d+).npy")
    print "Loading epoch {}: {}".format(epoch, file)
    z = np.load(file)
    z_depth = 10
    enc = cluster_agglomerative(z, z_depth)
    ds = load_dataset()
    write_encodings(enc, ds, output_path)


if __name__ == "__main__":
    main()
