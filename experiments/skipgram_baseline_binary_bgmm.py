import numpy as np
from discrete_skip_gram.clustering.cluster_train import train_cluster_battery
from discrete_skip_gram.clustering.gmm import validate_balanced_gmm


def main():
    bzks = [512, 256, 128, 64, 32]
    iters = 20
    z_k = 2
    output_path = "output/skipgram_baseline_binary_bgmm"
    cooccurrence = np.load('output/cooccurrence.npy')
    input_paths = ['output/skipgram_baseline', 'output/skipgram_baseline-l1', 'output/skipgram_baseline-l2']
    labels = ['baseline', 'l1', 'l2']
    train_cluster_battery(
        output_path=output_path,
        input_paths=input_paths,
        labels=labels,
        bzks=bzks,
        iters=iters,
        z_k=z_k,
        desc="Binary BGMM",
        cooccurrence=cooccurrence,
        val_fun=validate_balanced_gmm)


if __name__ == "__main__":
    main()
