from discrete_skip_gram.dataset import Dataset
from discrete_skip_gram.s2s_model import S2SModel
from discrete_skip_gram.prior import Prior
from discrete_skip_gram.corpora import reuters_words
from tqdm import tqdm
import os
import numpy as np
from discrete_skip_gram.regularization import l1l2


def write_generated(path, words):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        for word in words:
            f.write(word)
            f.write("\n")


def write_autoencoded(path, words, autoencodings):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        for data in zip(words, *autoencodings):
            f.write("{}: {}\n".format(data[0], ", ".join(data[1:])))


def experiment(path_generated, path_autoencoded, path_model, dataset,
               window=5, hidden_dim=256, nb_epoch=1000,
               nb_batch=64, batch_size=128, lr=1e-4,
               checkpoint_frequency=50, regularizer=None):
    prior = Prior(32, 5, 10)
    model = S2SModel(dataset.x_k, dataset.depth, prior.k, prior.maxlen, hidden_dim, lr=lr, regularizer=regularizer)
    test_size = 128
    autoencoded_size = 32
    for epoch in tqdm(range(nb_epoch), desc="Training"):
        generated_words = dataset.matrix_to_words(model.decode(prior.prior_samples(test_size)))
        write_generated(path_generated.format(epoch), generated_words)
        samples = dataset.sample_vectors(test_size)
        sample_words = dataset.matrix_to_words(samples)
        sample_autoencodings = [dataset.matrix_to_words(model.autoencode(samples)) for _ in range(autoencoded_size)]
        write_autoencoded(path_autoencoded.format(epoch), sample_words, sample_autoencodings)
        if epoch % checkpoint_frequency == 0:
            model.save(path_model.format(epoch))
        x_loss = []
        z_loss = []
        for _ in tqdm(range(nb_batch), desc="Epoch {}".format(epoch)):
            x, xnoised = dataset.sample_skip_grams(batch_size, window)
            z = prior.prior_samples(batch_size)
            xl, zl = model.train_batch(x, z, xnoised)
            x_loss.append(xl)
            z_loss.append(zl)
        x_loss = np.mean(x_loss, axis=None)
        z_loss = np.mean(z_loss, axis=None)
        tqdm.write("Epoch: {}, X loss: {}, Z loss: {}".format(epoch, x_loss, z_loss))


def main():
    path = "output/discrete_skip_gram"
    path_generated = os.path.join(path, "generated-{:08d}.txt")
    path_autoencoded = os.path.join(path, "autoencoded-{:08d}.txt")
    path_model = os.path.join(path, "model-{:08d}.h5")
    words = reuters_words()
    dataset = Dataset(words)
    window = 5
    experiment(path_generated, path_autoencoded, path_model,
               dataset=dataset,
               window=window,
               nb_batch=256,
               lr=1e-5,
               hidden_dim=1024,
               regularizer=l1l2(1e-6, 1e-6))


if __name__ == "__main__":
    main()
