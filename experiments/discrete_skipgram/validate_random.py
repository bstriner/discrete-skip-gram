import numpy as np

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.validation import write_validation

if __name__ == '__main__':
    x = load_cooccurrence('output/cooccurrence.npy')
    encoding_path = "output/random_encoding.npy"
    output_path = 'output/validate_random.csv'
    encoding = np.load(encoding_path)
    write_validation(output_path=output_path, encoding=encoding, co=x)
