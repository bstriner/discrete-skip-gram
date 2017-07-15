from discrete_skip_gram.skipgram.validation import validate_binary

if __name__ == '__main__':
    validate_binary('output/validate_random.csv', 'output/cooccurrence.npy', 'output/random_encoding.npy')
