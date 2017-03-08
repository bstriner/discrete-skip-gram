import os
from nltk.tokenize import word_tokenize
import pickle
import numpy as np


def clean_word(word):
    return "".join(c for c in word.lower() if ord(c) < 128)


def tokenize(text):
    return [clean_word(word) for word in word_tokenize(text) if clean_word(word)]


def make_path(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class WikiDataset(object):
    def __init__(self, wiki, path, min_length=20):
        self.wiki = wiki
        self.path = path
        self.min_length = min_length
        self.length = None
        self.charset = None
        self.ids = None
        self.charmap = None
        self.unique_words = None
        self.total_word_count = None

    def data_path(self):
        return os.path.join(self.path, "data.pkl")

    def log_path(self):
        return os.path.join(self.path, "log.txt")

    def doc_path(self, doc_id):
        return os.path.join(self.path,
                            'docs',
                            "{:03d}".format(doc_id / 1000000),
                            "{:03d}".format((doc_id % 1000000) / 1000),
                            "{:09d}.npy".format(doc_id))

    def write_log(self):
        with open(self.log_path(), 'w') as f:
            f.write("Data path: {}\n".format(self.data_path()))
            f.write("Doc count: {}\n".format(len(self.ids)))
            f.write("Total word count: {}\n".format(self.total_word_count))
            f.write("Unique word count: {}\n".format(len(self.unique_words)))
            f.write("Character count: {}\n".format(len(self.charset)))
            f.write("Longest word: {}\n".format(self.length))
            f.write("Charset: {}\n".format(", ".join("[{}]".format(c) for c in self.charset)))

    def process(self):
        self.preprocess()
        if not os.path.exists(self.log_path()):
            print("Processing")
            for doc in self.wiki.docs():
                if doc.id in self.ids:
                    print("Processing doc {}".format(doc.id))
                    mat = self.words_to_matrix(tokenize(doc.text))
                    with open(self.doc_path(doc.id), 'wb') as f:
                        np.save(f, mat)
            self.write_log()

    def word_to_vector(self, word):
        vec = [self.charmap[c] + 1 for c in word]
        while len(vec) < self.length:
            vec.append(0)
        return np.array(vec).reshape((1, -1)).astype(np.int32)

    def words_to_matrix(self, words):
        return np.vstack(self.word_to_vector(word) for word in words)

    def preprocess(self):
        if os.path.exists(self.data_path()):
            with open(self.data_path(), 'rb') as f:
                data = pickle.load(f)
        else:
            data = self.preprocess_data()
            make_path(self.data_path())
            with open(self.data_path(), 'wb') as f:
                pickle.dump(data, f)

        self.length, self.charset, self.ids, self.unique_words, self.total_word_count = data
        self.charmap = {c: i for i, c in enumerate(self.charset)}

    def preprocess_data(self):
        print("Preprocessing")
        ids = []
        charset = set()
        unique_words = set()
        total_word_count = 0
        length = 0
        for doc in self.wiki.docs():
            print("Preprocessing doc {}".format(doc.id))
            words = tokenize(doc.text)
            doc_len = len(words)
            if doc_len < self.min_length:
                print("Skipping doc {}, length {}".format(doc.id, doc_len))
            else:
                ids.append(doc.id)
                total_word_count += doc_len
                for word in words:
                    unique_words.add(word)
                    for char in word:
                        charset.add(char)
                        if len(word) > length:
                            length = len(word)
        charset = list(charset)
        charset.sort()
        ids.sort()
        unique_words = list(unique_words)
        unique_words.sort()
        return length, charset, ids, unique_words, total_word_count
