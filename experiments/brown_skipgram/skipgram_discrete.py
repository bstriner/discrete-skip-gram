#import os
#os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np

from discrete_skip_gram.skipgram_models.skipgram_discrete_model import SkipgramDiscreteModel
from sample_validation import validation_load
from discrete_skip_gram.layers.utils import leaky_relu
from dataset_util import load_dataset
from keras.regularizers import L1L2


def main():
    outputpath = "output/skipgram_discrete"
    ds = load_dataset()
    vd = validation_load()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 512
    window = 7
    frequency = 3
    units = 512
    embedding_units = 128
    z_k = 2
    z_depth = 10
    kernel_regularizer = L1L2(1e-7, 1e-7)
    embeddings_regularizer = L1L2(1e-7, 1e-7)
    lr = 1e-3
    lr_a = 1e-3
    adversary_weight = 1e-2
    layernorm = False
    model = SkipgramDiscreteModel(dataset=ds,
                                  z_k=z_k,
                                  z_depth=z_depth,
                                  window=window,
                                  embedding_units=embedding_units,
                                  kernel_regularizer=kernel_regularizer,
                                  embeddings_regularizer=embeddings_regularizer,
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
