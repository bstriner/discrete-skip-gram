# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

import numpy as np

from dataset import load_dataset
from discrete_skip_gram.models.word_skipgram_baseline_relu import WordSkipgramBaselineRelu
import theano.tensor as T


# 44sec on laptop w 512 units

def leaky_relu(x):
    return T.nnet.relu(x, 0.2)


# maybe try movie_reviews or reuters
def main():
    outputpath = "output/brown/skipgram_baseline_relu_layernorm"
    dataset = load_dataset()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 512
    frequency = 50
    window = 2
    units = 512
    embedding_units = 128
    lr = 1e-3

    model = WordSkipgramBaselineRelu(dataset=dataset,
                                     window=window,
                                     inner_activation=leaky_relu,
                                     units=units,
                                     embedding_units=embedding_units,
                                     layernorm=True,
                                     lr=lr)
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
