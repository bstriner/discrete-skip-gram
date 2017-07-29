import numpy as np

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.util import latest_file
from discrete_skip_gram.skipgram.validation import validate_encoding_flat, validate_empty

if __name__ == '__main__':
    output_path = "output/skipgram_flat_validate.txt"
    input_path = "output/skipgram_flat-balance"
    cooccurrence = load_cooccurrence('output/cooccurrence.npy')
    z_k = 1024

    val = validate_encoding_flat(cooccurrence=cooccurrence)
    encoding_path, epoch = latest_file(input_path, "encodings-(\d+).npy")
    if not epoch:
        raise ValueError("No file found at {}".format(input_path))
    print("Epoch {}: {}".format(epoch, encoding_path))
    enc = np.load(encoding_path)
    nll = val(enc, z_k)
    with open(output_path, 'w') as f:
        f.write("NLL: {}".format(nll))
    print("NLL: {}".format(nll))
    validate_empty(enc, z_k)


