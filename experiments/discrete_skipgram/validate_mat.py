import numpy as np

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.validation import validate
from discrete_skip_gram.skipgram.util import latest_file
if __name__ == '__main__':
    x = load_cooccurrence('output/cooccurrence.npy')
    encoding_path, epoch = latest_file("output/skipgram_discrete_mat","encodings-(\d+).npy")
    print("Epoch {}: {}".format(epoch, encoding_path))
    validate('output/validate_mat.csv', 'output/cooccurrence.npy', encoding_path)

