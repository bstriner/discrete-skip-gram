from train_discrete_skip_gram import experiment
from discrete_skip_gram.dataset import Dataset
from discrete_skip_gram.corpora import reuters_words
from discrete_skip_gram.prior import Prior
from discrete_skip_gram.regularization import l1l2
import os


def s2smain(path, **kwargs):
    path_generated = os.path.join(path, "generated-{:08d}.txt")
    path_autoencoded = os.path.join(path, "autoencoded-{:08d}.txt")
    path_model = os.path.join(path, "model-{:08d}.h5")
    words = reuters_words()
    dataset = Dataset(words)
    window = 7
    experiment(path_generated, path_autoencoded, path_model,
               prior=Prior(64, 5, 10),
               dataset=dataset,
               window=window,
               nb_batch=256,
               lr=1e-4,
               hidden_dim=1024,
               regularizer=l1l2(1e-8, 1e-8),
               checkpoint_frequency=25,
               batch_size=128,
               nb_epoch=101,
               **kwargs)


def main():
    path = "output/s2s"
    s2smain(os.path.join(path, "adversarial"),
            encode_deterministic=False, decode_deterministic=False,
            adversarial_x=True, adversarial_z=True)
    s2smain(os.path.join(path, "encode_deterministic"),
            encode_deterministic=True, decode_deterministic=False,
            adversarial_x=False, adversarial_z=False)
    s2smain(os.path.join(path, "decode_deterministic"),
            encode_deterministic=False, decode_deterministic=True,
            adversarial_x=False, adversarial_z=False)


if __name__ == "__main__":
    main()
