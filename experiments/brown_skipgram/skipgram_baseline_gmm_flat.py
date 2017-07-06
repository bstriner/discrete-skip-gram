# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

from discrete_skip_gram.skipgram_models.skipgram_validation_flat_model import validate_skipgram_flat


def main():
    outputpath = "output/skipgram_baseline_gmm_flat"
    embeddingpath = "output/cluster_gmm/encodings.npy"
    validate_skipgram_flat(outputpath=outputpath, embeddingpath=embeddingpath)


if __name__ == "__main__":
    main()
