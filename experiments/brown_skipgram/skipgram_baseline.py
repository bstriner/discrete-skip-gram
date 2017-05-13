#import os
#os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np

from discrete_skip_gram.skipgram_models.skipgram_baseline_model import SkipgramBaselineModel
from sample_validation import validation_load
from discrete_skip_gram.layers.utils import leaky_relu
from dataset_util import load_dataset
from keras.regularizers import L1L2


def main():
    outputpath = "output/skipgram_baseline"
    ds = load_dataset()
    vd = validation_load()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 1024
    window = 2
    frequency = 50
    units = 512
    embedding_units = 128
    kernel_regularizer = L1L2(1e-9, 1e-9)
    embeddings_regularizer = L1L2(1e-9, 1e-9)
    lr = 1e-3
    layernorm = False
    model = SkipgramBaselineModel(dataset=ds,
                                  window=window,
                                  embedding_units=embedding_units,
                                  kernel_regularizer=kernel_regularizer,
                                  embeddings_regularizer=embeddings_regularizer,
                                  layernorm=layernorm,
                                  inner_activation=leaky_relu,
                                  units=units,
                                  lr=lr)
    model.summary()
    vn = 4096
    validation_data = ([vd[0][:vn, ...], vd[1][:vn, ...]], np.ones((vn, 1), dtype=np.float32))

    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_data,
                output_path=outputpath,
                frequency=frequency)


if __name__ == "__main__":
    main()
