from discrete_skip_gram.skipgram.validation import validate

if __name__ == '__main__':
    validate('output/validate_gmm.csv', 'output/cooccurrence.npy', 'output/cluster_gmm/encodings.npy')
