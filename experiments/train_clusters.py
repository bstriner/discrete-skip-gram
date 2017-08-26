import os
import sys

sys.path.append(os.path.dirname(__file__))
import skipgram_baseline_binary_gmm
import skipgram_baseline_binary_kmeans
import skipgram_baseline_binary_bgmm
import skipgram_baseline_flat_gmm
import skipgram_baseline_flat_kmeans
import skipgram_baseline_flat_bgmm


def main():
    skipgram_baseline_binary_gmm.main()
    skipgram_baseline_binary_kmeans.main()
    skipgram_baseline_binary_bgmm.main()
    skipgram_baseline_flat_gmm.main()
    skipgram_baseline_flat_kmeans.main()
    skipgram_baseline_flat_bgmm.main()


if __name__ == "__main__":
    main()
