# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

import numpy as np
from keras.regularizers import L1L2

from dataset import load_dataset
from discrete_skip_gram.models.word_skipgram_test import WordSkipgramTest


# maybe try movie_reviews or reuters
def main():
    outputpath = "output/brown/skipgram_baseline_discrete"
    embedding_file = "{}/encoding.npy".format(outputpath)
    embedding = np.load(embedding_file)
    dataset = load_dataset()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 512
    frequency = 10
    window = 2
    units = 128
    lr = 1e-3
    kernel_regularizer = L1L2(1e-7, 1e-7)
    z_k = 2

    model = WordSkipgramTest(dataset=dataset,
                             embedding=embedding,
                             window=window,
                             z_k=z_k,
                             kernel_regularizer=kernel_regularizer,
                             units=units, lr=lr)
    validation_n = 4096
    vd = dataset.cbow_batch(n=validation_n, window=window, test=True)

    model.train(batch_size=batch_size,
                epochs=epochs,
                frequency=frequency,
                steps_per_epoch=steps_per_epoch,
                validation_data=([vd[1], vd[0]], np.ones((validation_n, 1), dtype=np.float32)),
                output_path=outputpath)


if __name__ == "__main__":
    main()