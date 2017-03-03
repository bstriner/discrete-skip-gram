from discrete_skip_gram.dataset import Dataset
from discrete_skip_gram.s2s_model import S2SModel
from discrete_skip_gram.prior import Prior
from discrete_skip_gram.corpora import shakespeare_words_short
from tqdm import tqdm
import os
import numpy as np
from train_discrete_skip_gram import experiment
from discrete_skip_gram.regularization import l1l2


def main():
    path = "output/discrete_skip_gram_short"
    path_generated = os.path.join(path, "generated-{:08d}.txt")
    path_autoencoded = os.path.join(path, "autoencoded-{:08d}.txt")
    path_model = os.path.join(path, "model-{:08d}.h5")
    words = shakespeare_words_short()
    dataset = Dataset(words)
    window = 0
    experiment(path_generated, path_autoencoded, path_model,
               prior=Prior(16, 5, 10),
               dataset=dataset,
               window=window,
               nb_batch=256,
               lr=3e-4,
               hidden_dim=1024,
               regularizer=l1l2(1e-8, 1e-8),
               encode_deterministic=True,
               decode_deterministic=False)


if __name__ == "__main__":
    main()
