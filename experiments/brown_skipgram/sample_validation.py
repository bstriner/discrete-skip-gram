import os
import pickle

from discrete_skip_gram.dataset_util import load_dataset


def sample_data():
    print "Loading"
    dataset = load_dataset()
    window = 2
    validation_n = 65536
    print "Sampling"
    vd = dataset.skip_gram_batch(n=validation_n, window=window, test=True)
    return [vd[1], vd[0]]


validation_path = "output/validation.pkl"


def validation_save(vd):
    print "Saving"
    if os.path.exists(validation_path):
        raise ValueError("Path already exists: {}".format(validation_path))
    if not os.path.exists(os.path.dirname(validation_path)):
        os.makedirs(os.path.dirname(validation_path))
    with open(validation_path, 'wb') as f:
        pickle.dump(vd, f)


def validation_load():
    with open(validation_path, 'rb') as f:
        return pickle.load(f)


def main():
    vd = sample_data()
    validation_save(vd)


if __name__ == "__main__":
    main()
