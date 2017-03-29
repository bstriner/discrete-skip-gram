from nltk.corpus import brown
import itertools
from nltk.stem.porter import PorterStemmer
import numpy as np

from discrete_skip_gram.datasets.utils import clean_docs, simple_clean, get_words
from discrete_skip_gram.datasets.character_dataset import brown_docs, get_charset, docs_to_arrays


def main():
    docs = brown_docs()
    docs = clean_docs(docs, simple_clean)
    count = sum(len(doc) for doc in docs)
    words = get_words(docs)
    charset, charmap = get_charset(words)
    print "Words: {}, Unique: {}".format(count, len(words))
    print docs[0]
    print docs[1]
    adocs = docs_to_arrays(docs, charmap)


if __name__ == "__main__":
    main()
