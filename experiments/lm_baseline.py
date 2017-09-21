import numpy as np
from discrete_skip_gram.util import generate_ngrams
def main():
    corpus=np.load('output/corpus/corpus.npy')
    embeddings=np.load("output/skipgram_baseline/z-512-embeddings.npy")
    enc = embeddings[0,:,:] # 4946, 512
    n = 5
    ng = generate_ngrams(corpus, 5)



if __name__ == "__main__":
    main()