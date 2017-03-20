import os

# os.environ["THEANO_FLAGS"] = "device=cpu"
from tqdm import tqdm
import theano
import theano.tensor as T
from keras.initializers import glorot_uniform, zeros
import numpy as np
from discrete_skip_gram.constraints import ClipConstraint
from discrete_skip_gram.ddpg import DDPG
from nltk.corpus import shakespeare
import itertools


def clean_word(word):
    return [c for c in word if ord(c) < 128]


def clean_words(words):
    return [clean_word(w) for w in words if clean_word(w)]


def shakespeare_words():
    return itertools.chain.from_iterable(shakespeare.words(f) for f in shakespeare.fileids())


def get_charset(words):
    """
    List unique characters
    :param words:
    :return: list of characters, dictionary from characters to indexes
    """
    charset = list(set(itertools.chain.from_iterable(words)))
    charset.sort()
    charmap = {c: i for i, c in enumerate(charset)}
    return charset, charmap


def map_word(word, charmap):
    """
    Convert string to list of indexes into charset
    :param word:
    :param charmap:
    :return:
    """
    return [charmap[c] for c in word]


def map_words(words, charmap):
    return [map_word(w, charmap) for w in words]


def vector_to_matrix(vector, depth):
    ar = [c + 1 for c in vector]
    while len(ar) < depth:
        ar.append(0)
    return np.array(ar).astype(np.int32).reshape(1, -1)


def vectors_to_matrix(vectors, depth):
    return np.vstack(vector_to_matrix(v, depth) for v in vectors)


def decode_row(row, charset):
    """
    Output vector to a string
    :param row:
    :param charset:
    :return:
    """
    return "".join([charset[x - 1] if x > 0 else " " for x in row])


def decode_output(output, charset):
    """
    Output matrix to list of strings
    :param output:
    :param charset:
    :return:
    """
    return [decode_row(row, charset) for row in output]


def write_results(epoch, x_gen, charset):
    path = "output/ddpg/epoch-{}.txt".format(epoch)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        for word in decode_output(x_gen, charset):
            f.write(word)
            f.write("\n")
            print(word)


def main():
    latent_dim = 100
    #generator_hidden_dim = 256
    #discriminator_hidden_dim = 64
    epochs = 100
    batches = 256
    batch_size = 128

    words = clean_words(shakespeare_words())
    depth = max(len(w) for w in words) + 1
    charset, charmap = get_charset(words)
    vectors = map_words(words, charmap)
    mat = vectors_to_matrix(vectors, depth)
    x_k = len(charset)
    m = DDPG(x_k=x_k, latent_dim=latent_dim)
        #, generator_hidden_dim=generator_hidden_dim,
        #     discriminator_hidden_dim=discriminator_hidden_dim)

    for epoch in tqdm(range(epochs)):
        for _ in tqdm(range(batches)):
            idx = np.random.randint(0, mat.shape[0], (batch_size,))
            x_real = mat[idx, :]
            loss, dloss, gloss, ploss = m.train(x_real)
            tqdm.write("Loss: {} (D: {}, G: {}, P: {})".format(loss, dloss, gloss, ploss))
        x_gen = m.predict(batch_size, depth)
        write_results(epoch=epoch, x_gen=x_gen, charset=charset)


if __name__ == "__main__":
    main()
