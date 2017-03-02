from nltk.corpus import reuters, shakespeare
import itertools


def clean_word(word):
    return "".join(c for c in word.lower() if ord(c) < 128)


def clean(words):
    return [w for w in(clean_word(word) for word in words) if len(w)>0]


def reuters_words():
    return clean(reuters.words())


def shakespeare_words():
    return clean(itertools.chain.from_iterable(shakespeare.words(fileid) for fileid in shakespeare.fileids()))


def shakespeare_words_short(maxlen = 7):
    return [word for word in shakespeare_words() if len(word) <= maxlen]
