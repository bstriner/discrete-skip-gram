# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

from discrete_skip_gram.skipgram_models.skipgram_validation_model import validate_skipgram


def main():
    outputpath = "output/skipgram_baseline_gmm"
    embeddingpath = "output/cluster_gmm/encodings.npy"
    validate_skipgram(outputpath=outputpath, embeddingpath=embeddingpath)


if __name__ == "__main__":
    main()
