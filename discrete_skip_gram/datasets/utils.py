from nltk.stem.porter import PorterStemmer
from collections import Counter
import itertools
import numpy as np


def stem_word():
    stemmer = PorterStemmer()

    def fun(w):
        return "".join(c for c in stemmer.stem(w).lower() if ord(c) < 128)

    return fun


def simple_clean(w):
    return "".join(c for c in w.lower() if ord('a') <= ord(c) <= ord('z'))


def clean_words(ws, clean_word):
    return [clean_word(w) for w in ws if clean_word(w)]


def clean_docs(docs, clean_word):
    return [clean_words(doc, clean_word) for doc in docs]


def get_words(docs):
    """
    Get unique words from a set of docs (list of list of words)
    :param docs: 
    :return: 
    """
    words = list(set(itertools.chain.from_iterable(docs)))
    words.sort()
    return words


def count_words(docs):
    """
    count word occurences
    :param docs: 
    :return: 
    """
    return Counter(itertools.chain.from_iterable(docs))


def format_encoding(enc):
    return "".join(chr(ord('a') + enc[i]) for i in range(enc.shape[0]))


def format_encoding_sequential_continuous(enc):
    assert (len(enc.shape) == 2)
    k = enc.shape[1]
    depth = enc.shape[0]
    fmt = []
    for i in range(depth):
        e = enc[i, :]
        s = np.greater(e, 0).astype(np.int32)
        t = np.sum(np.power(2, np.arange(k)) * s)
        fmt.append(chr(ord('a') + t))
    return "".join(fmt)
