import pickle
import os

from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.datasets.utils import clean_docs, simple_clean
from discrete_skip_gram.datasets.word_dataset import WordDataset

_brown_path = "output/brown/corpus.pkl"

def load_dataset():
    with open(_brown_path, 'rb') as f:
        return pickle.load(f)


def save_dataset(ds):
    if os.path.exists(_brown_path):
        raise ValueError("Already exists: {}".format(_brown_path))
    if not os.path.exists(os.path.dirname(_brown_path)):
        os.makedirs(os.path.dirname(_brown_path))
    with open(_brown_path, 'wb') as f:
        pickle.dump(ds, f)


def main():
    min_count = 5
    docs = clean_docs(brown_docs(), simple_clean)
    docs, tdocs = docs[:-5], docs[-5:]
    ds = WordDataset(docs, min_count, tdocs=tdocs)
    ds.summary()
    save_dataset(ds)


if __name__ == "__main__":
    main()
