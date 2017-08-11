import numpy as np
from discrete_skip_gram.clustering.cluster_train import cluster_dir
from discrete_skip_gram.clustering.kmeans import validate_cluster_km


def main():
    bzks = [512, 256, 128, 64, 32]
    iters = 5
    z_k = 1024
    output_path = "output/skipgram_baseline_flat_kmeans.npz"
    cooccurrence = np.load('output/cooccurrence.npy')
    baseline = cluster_dir(input_path='output/skipgram_baseline',
                           bzks=bzks,
                           iters=iters,
                           z_k=z_k,
                           cooccurrence=cooccurrence,
                           val_fun=validate_cluster_km)
    l1 = cluster_dir(input_path='output/skipgram_baseline-l1',
                     bzks=bzks,
                     iters=iters,
                     z_k=z_k,
                     cooccurrence=cooccurrence,
                     val_fun=validate_cluster_km)
    l2 = cluster_dir(input_path='output/skipgram_baseline-l2',
                     bzks=bzks,
                     iters=iters,
                     z_k=z_k,
                     cooccurrence=cooccurrence,
                     val_fun=validate_cluster_km)
    # (bzks, biters, iters)
    np.savez(output_path, baseline=baseline, l1=l1, l2=l2)


if __name__ == "__main__":
    main()
