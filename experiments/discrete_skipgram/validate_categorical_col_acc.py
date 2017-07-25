from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.util import latest_file
from discrete_skip_gram.skipgram.validation import validate_flat

if __name__ == '__main__':
    x = load_cooccurrence('output/cooccurrence.npy')
    encoding_path, epoch = latest_file("output/skipgram_categorical_col_acc", "encodings-(\d+).npy")
    if not epoch:
        raise ValueError("No file found")
    print("Epoch {}: {}".format(epoch, encoding_path))
    validate_flat('output/validate_categorical_col_acc.csv', 'output/cooccurrence.npy', encoding_path, z_k=1024)
