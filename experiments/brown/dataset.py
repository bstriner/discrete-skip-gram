import pickle
import os

from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.datasets.utils import clean_docs, simple_clean
from discrete_skip_gram.datasets.word_dataset import WordDataset

# todo: remove numbers
# min = 5
# Total wordcount: 1149063
# Unique words: 49364, Filtered: 14121
# min = 20
# Total wordcount: 1149063
# Unique words: 49364, Filtered: 4975
# Removing punctuation
# Total wordcount: 995033
# Unique words: 45837, Filtered: 4908

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


def create_dataset():
    min_count = 20
    docs = clean_docs(brown_docs(), simple_clean)
    docs, tdocs = docs[:-5], docs[-5:]
    ds = WordDataset(docs, min_count, tdocs=tdocs)
    ds.summary()
    save_dataset(ds)


def tst_dataset():
    ds = load_dataset()
    ds.summary()


def main():
    create_dataset()
    tst_dataset()


if __name__ == "__main__":
    main()
