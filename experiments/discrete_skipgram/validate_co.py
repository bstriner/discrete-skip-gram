from discrete_skip_gram.skipgram.util import latest_file
from discrete_skip_gram.skipgram.validation import validate

if __name__ == '__main__':
    encoding_path, epoch = latest_file("output/skipgram_discrete_co", "encodings-(\d+).npy")
    print("Epoch {}: {}".format(epoch, encoding_path))
    validate('output/validate_co.csv', 'output/cooccurrence.npy', encoding_path)
