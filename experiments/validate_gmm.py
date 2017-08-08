from discrete_skip_gram.flat_validation import validate_binary

if __name__ == '__main__':
    validate_binary('output/validate_gmm.csv', 'output/cooccurrence.npy', 'output/cluster_gmm/encodings.npy')
