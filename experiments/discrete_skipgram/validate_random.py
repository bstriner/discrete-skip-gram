from discrete_skip_gram.skipgram.validation import validate

if __name__ == '__main__':
    validate('output/validate_random.csv', 'output/cooccurrence.npy', 'output/random_encoding.npy')
