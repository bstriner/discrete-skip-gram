from nltk.corpus import reuters, shakespeare
import itertools

def clean_word(word):
    return "".join(c for c in word.lower() if ord(c) < 128)


def clean(words):
    return [clean_word(word) for word in words]


def reuters_words():
    return clean(reuters.words())
def shakespeare_words():
    return clean(itertools.chain.from_iterable(shakespeare.words(fileid) for fileid in shakespeare.fileids))
