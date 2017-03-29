from nltk.stem.porter import PorterStemmer

import itertools


def stem_word():
    stemmer = PorterStemmer()

    def fun(w):
        return "".join(c for c in stemmer.stem(w).lower() if ord(c) < 128)

    return fun


def simple_clean(w):
    return "".join(c for c in w.lower() if ord(c) < 128)


def clean_words(ws, clean_word):
    return [clean_word(w) for w in ws if clean_word(w)]


def clean_docs(docs, clean_word):
    return [clean_words(doc, clean_word) for doc in docs]


def get_words(docs):
    words = list(set(itertools.chain.from_iterable(docs)))
    words.sort()
    return words
