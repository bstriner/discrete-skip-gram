import os
# os.environ["THEANO_FLAGS"]="optimizer=None"
import csv

from nltk.corpus import brown
import itertools
from nltk.stem.porter import PorterStemmer
import numpy as np

from discrete_skip_gram.datasets.utils import clean_docs, simple_clean, get_words, count_words, format_encoding
from discrete_skip_gram.datasets.word_dataset import docs_to_arrays, skip_gram_generator, skip_gram_batch
from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.callbacks.write_encodings import WriteEncodings
from discrete_skip_gram.models.word_dqn import WordDQN


def main():
    min_count = 20
    z_depth = 5
    z_k = 4
    units = 256
    batch_size = 64
    window = 7
    samples = 1
    batches = 256
    decoder_batches = 4
    value_batches = 4
    epochs = 1000

    docs = brown_docs()
    docs = clean_docs(docs, simple_clean)
    count = sum(len(doc) for doc in docs)
    wordcounts = count_words(docs)
    wordset = [k for k, v in wordcounts.iteritems() if v >= min_count]
    wordset.sort()
    wordmap = {w: i for i, w in enumerate(wordset)}
    adocs = docs_to_arrays(docs, wordmap)

    k = len(wordset) + 1

    print "Total wordcount: {}".format(count)
    print "Unique words: {}, Filtered: {}".format(len(wordcounts), len(wordset))

    model = WordDQN(z_depth, z_k, x_k=k, y_k=k, units=units, discount=0.8, initial_exploration=0.3)
    model.summary()

    def callback(epoch):
        path = "output/brown_word_dqn/epoch-{:08d}.txt".format(epoch)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            x, y = skip_gram_batch(adocs, 0, batch_size)
            zs = [model.model_encoder.predict_on_batch(x) for _ in range(16)]
            for i in range(batch_size):
                w = wordset[x[i - 1, 0]] if i > 0 else "__UNK__"
                strs = ", ".join(format_encoding(z[i]) for z in zs)
                f.write("{}: {}\n".format(w, strs))
        if (epoch + 1) % 10 == 0:
            path = "output/brown_word_dqn/encodings-{:08}.csv".format(epoch)
            words = np.arange(0, k).reshape((-1, 1))
            encodings = model.model_encoder_deterministic.predict(words, batch_size=batch_size)
            with open(path, 'w') as f:
                w = csv.writer(f)
                w.writerow(["Index", "Word", "Encoding"])
                for i in range(k):
                    if i == 0:
                        word = "_UNK_"
                    else:
                        word = wordset[i - 1]
                    encoding = format_encoding(encodings[i, :])
                    w.writerow([i, word, encoding])

    gen = skip_gram_generator(adocs, window, batch_size)
    model.fit_generator(gen,
                        epochs=epochs,
                        callback=callback,
                        samples=samples,
                        batches=batches,
                        decoder_batches=decoder_batches,
                        value_batches=value_batches)


if __name__ == "__main__":
    main()
