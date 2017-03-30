import os
#os.environ["THEANO_FLAGS"]="optimizer=None"

from nltk.corpus import brown
import itertools
from nltk.stem.porter import PorterStemmer
import numpy as np

from discrete_skip_gram.datasets.utils import clean_docs, simple_clean, get_words
from discrete_skip_gram.datasets.character_dataset import get_charset, docs_to_arrays, skip_gram_generator
from discrete_skip_gram.models.character_skip_gram import CharacterSkipGram
from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.callbacks.write_encodings import WriteEncodings

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
    x_k = len(charset)
    units = 256
    latent_depth = 8
    latent_k = 4
    window = 7
    batch_size = 64
    epochs = 1000
    steps_per_epoch = 256
    model = CharacterSkipGram(latent_depth, latent_k, x_k, x_k, units)
    model.model.summary()
    cb = WriteEncodings(model.encoder, adocs, charset, "output/brown_character_skipgram/encoded-{:08d}.txt")
    #cb.on_epoch_end(0,None)
    model.model.fit_generator(skip_gram_generator(adocs, window, batch_size), epochs=epochs,
                              steps_per_epoch=steps_per_epoch, callbacks=[cb])


if __name__ == "__main__":
    main()
