import numpy as np
from discrete_skip_gram.skipgram.util import latest_file, write_encodings
from discrete_skip_gram.skipgram.validation import validate_binary

from discrete_skip_gram.corpus import load_corpus

if __name__ == '__main__':
    encoding_path, epoch = latest_file("output/skipgram_discrete_sum_acc", "encodings-(\d+).npy")
    print("Epoch {}: {}".format(epoch, encoding_path))
    validate_binary('output/validate_discrete_sum_acc.csv', 'output/cooccurrence.npy', encoding_path)
    corpus_path = "output/corpus"
    vocab, corpus = load_corpus(corpus_path)
    encodings = np.load(encoding_path)
    write_encodings("output/validate_discrete_sum_acc-encodings.csv",vocab=vocab, encodings=encodings)

