from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.validation import validate_flat, validate_encoding_flat

from discrete_skip_gram.util import latest_file

if __name__ == '__main__':
    x = load_cooccurrence('output/cooccurrence.npy')
    val = validate_encoding_flat(cooccurrence=x)
    encoding_path, epoch = latest_file("output/skipgram_categorical_col", "encodings-(\d+).npy")
    if not epoch:
        raise ValueError("No file found")
    print("Epoch {}: {}".format(epoch, encoding_path))
    validate_flat('output/validate_categorical_col.csv', 'output/cooccurrence.npy', encoding_path, z_k=1024)
