from discrete_skip_gram.dataset import write_dataset

if __name__ == '__main__':
    write_dataset('output/dataset.npy', 'output/corpus', window=3)
