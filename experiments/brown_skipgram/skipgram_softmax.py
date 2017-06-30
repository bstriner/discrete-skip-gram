# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np

from discrete_skip_gram.dataset_util import load_dataset
from discrete_skip_gram.layers.utils import leaky_relu
from discrete_skip_gram.skipgram_models.skipgram_softmax_model import SkipgramSoftmaxModel
from keras.regularizers import L1L2
from sample_validation import validation_load


def main():
    outputpath = "output/skipgram_softmax"
    ds = load_dataset()
    vd = validation_load()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 5
    units = 512
    embedding_units = 128
    z_k = 2
    z_depth = 10
    kernel_regularizer = L1L2(1e-8, 1e-8)
    embeddings_regularizer = L1L2(1e-8, 1e-8)
    lr = 3e-4
    lr_a = 1e-3
    loss_weight = 1e-2
    adversary_weight = 1e-4
    layernorm = False
    model = SkipgramSoftmaxModel(dataset=ds,
                                 z_k=z_k,
                                 z_depth=z_depth,
                                 window=window,
                                 embedding_units=embedding_units,
                                 kernel_regularizer=kernel_regularizer,
                                 embeddings_regularizer=embeddings_regularizer,
                                 loss_weight=loss_weight,
                                 adversary_weight=adversary_weight,
                                 lr_a=lr_a,
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
