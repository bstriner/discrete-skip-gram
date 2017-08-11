import numpy as np
from discrete_skip_gram.clustering.cluster_train import cluster_dir
from discrete_skip_gram.clustering.gmm import validate_balanced_gmm


def main():
    bzks = [512, 256, 128, 64, 32]
    iters = 5
    z_k = 2
    output_path = "output/skipgram_baseline_binary_bgmm.npz"
    cooccurrence = np.load('output/cooccurrence.npy')
    kwdata = {'bzks': np.array(bzks)}
    kwdata['baseline'] = cluster_dir(input_path='output/skipgram_baseline',
                                     bzks=bzks,
                                     iters=iters,
                                     z_k=z_k,
                                     cooccurrence=cooccurrence,
                                     val_fun=validate_balanced_gmm)
    kwdata['l1'] = cluster_dir(input_path='output/skipgram_baseline-l1',
                               bzks=bzks,
                               iters=iters,
                               z_k=z_k,
                               cooccurrence=cooccurrence,
                               val_fun=validate_balanced_gmm)
    kwdata['l2'] = cluster_dir(input_path='output/skipgram_baseline-l2',
                               bzks=bzks,
                               iters=iters,
                               z_k=z_k,
                               cooccurrence=cooccurrence,
                               val_fun=validate_balanced_gmm)
    # (bzks, biters, iters)
    np.savez(output_path, **kwdata)


if __name__ == "__main__":
    main()
