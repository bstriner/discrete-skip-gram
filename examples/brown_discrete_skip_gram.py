from discrete_skip_gram.train_discrete_skip_gram import train_discrete_skip_gram
from nltk.corpus import brown
from discrete_skip_gram.dataset import DatasetFiles
from discrete_skip_gram.prior import Prior
#Dataset words: 1161192, unique words: 49815, characters: 57, max length: 33, file count: 500

def main():
    dataset = DatasetFiles(brown)
    dataset.summary()
    path_generated = "output/brown/generated-{:09d}.txt"
    path_autoencoded = "output/brown/autoencoded-{:09d}.txt"
    path_encoded = "output/brown/encoded-{:09d}.txt"
    path_model = "output/brown/model-{:09d}.h5"
    prior = Prior(4, 3, 6)

    train_discrete_skip_gram(path_generated, path_autoencoded, path_encoded, path_model, dataset, prior,
               window=7, hidden_dim=512, nb_epoch=1001,
               nb_batch=256, batch_size=128, lr=3e-5,
               checkpoint_frequency=50, regularizer=None,
               encode_deterministic=False, decode_deterministic=False,
               adversarial_x=False, adversarial_z=False,
                             decay_z=0.0005)

if __name__ == "__main__":
    main()