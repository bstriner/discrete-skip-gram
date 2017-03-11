from discrete_skip_gram.dataset import Dataset
from discrete_skip_gram.s2s_model import S2SModel
from discrete_skip_gram.prior import Prior
from discrete_skip_gram.corpora import reuters_words
from discrete_skip_gram.train_discrete_skip_gram import train_discrete_skip_gram
from tqdm import tqdm
import os
import numpy as np
from discrete_skip_gram.regularization import l1l2


def main():
    path = "output/discrete_skip_gram"
    path_generated = os.path.join(path, "generated-{:08d}.txt")
    path_autoencoded = os.path.join(path, "autoencoded-{:08d}.txt")
    path_model = os.path.join(path, "model-{:08d}.h5")
    words = reuters_words()
    dataset = Dataset(words)
    window = 5
    train_discrete_skip_gram(path_generated, path_autoencoded, path_model,
               prior=Prior(64, 5, 10),
               dataset=dataset,
               window=window,
               nb_batch=256,
               lr=1e-5,
               hidden_dim=1024,
               regularizer=l1l2(1e-6, 1e-6))


if __name__ == "__main__":
    main()
