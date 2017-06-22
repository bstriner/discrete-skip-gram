#import os
#os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"

from discrete_skip_gram.skipgram_models.skipgram_validation_model import SkipgramValidationModel
from discrete_skip_gram.layers.utils import leaky_relu
from dataset_util import load_dataset
from keras.regularizers import L1L2
import numpy as np
from discrete_skip_gram.models.util import latest_model


def main():
    outputpath = "output/skipgram_discrete_binary_balanced_validate"
    inputpath = "output/skipgram_discrete_binary_balanced"
    embeddingpath, epoch = latest_model(inputpath, "encodings-(\\d+).npy")
    embedding = np.load(embeddingpath)
    print "Using epoch {}: {}".format(epoch, embeddingpath)
    ds = load_dataset()
    batch_size = 256
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 10
    units = 512
    embedding_units = 128
    z_k = 2
    kernel_regularizer = L1L2(1e-8, 1e-8)
    embeddings_regularizer = L1L2(1e-8, 1e-8)
    loss_weight = 1e-2
    lr = 3e-4
    layernorm = False
    batchnorm = True
    model = SkipgramValidationModel(dataset=ds,
                                    z_k=z_k,
                                    embedding=embedding,
                                    window=window,
                                    embedding_units=embedding_units,
                                    kernel_regularizer=kernel_regularizer,
                                    embeddings_regularizer=embeddings_regularizer,
                                    loss_weight=loss_weight,
                                    layernorm=layernorm,
                                    batchnorm=batchnorm,
                                    inner_activation=leaky_relu,
                                    units=units,
                                    lr=lr)
    model.summary()
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                output_path=outputpath,
                frequency=frequency)


if __name__ == "__main__":
    main()
