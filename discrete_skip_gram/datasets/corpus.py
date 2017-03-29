from nltk.corpus import brown

def brown_docs():
    return [list(brown.words(fileid)) for fileid in brown.fileids()]
