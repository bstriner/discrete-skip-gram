# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

import numpy as np
from keras.regularizers import L1L2

from dataset import load_dataset
from discrete_skip_gram.models.word_skipgram_sequential_sigmoid import WordSkipgramSequentialSigmoid
from sample_validation import validation_load
from discrete_skip_gram.layers.utils import leaky_relu


def main():
    outputpath = "output/brown/skipgram_sequential_sigmoid"
    dataset = load_dataset()
    vd = validation_load()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 512
    frequency = 20
    kernel_regularizer = L1L2(1e-9, 1e-9)
    window = 2
    units = 512
    embedding_units = 128
    z_depth = 8
    lr = 1e-3
    lr_a = 1e-3
    decay = 0.9
    hidden_layers = 2
    adversary_weight = 1.0
    schedule = np.power(decay, np.arange(z_depth))
    model = WordSkipgramSequentialSigmoid(dataset=dataset,
                                          adversary_weight=adversary_weight,
                                          schedule=schedule,
                                          hidden_layers=hidden_layers,
                                          embedding_units=embedding_units,
                                          z_depth=z_depth,
                                          window=window,
                                          inner_activation=leaky_relu,
                                          kernel_regularizer=kernel_regularizer,
                                          units=units, lr=lr, lr_a=lr_a)
    model.summary()
    vn = 4096
    model.train(batch_size=batch_size,
                epochs=epochs,
                frequency=frequency,
                steps_per_epoch=steps_per_epoch,
                validation_data=([vd[0][:vn], vd[1][:vn]], np.ones((vn, 1), dtype=np.float32)),
                output_path=outputpath)


if __name__ == "__main__":
    main()
