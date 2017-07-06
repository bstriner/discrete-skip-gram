from discrete_skip_gram.skipgram.cooccurrence import write_cooccurrence

if __name__ == '__main__':
    write_cooccurrence('output/cooccurrence.npy', 'output/dataset.npy', 'output/corpus')
