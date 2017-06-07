# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np

from dataset_util import load_dataset
from discrete_skip_gram.layers.utils import leaky_relu
from discrete_skip_gram.skipgram_models.skipgram_discrete_model import SkipgramDiscreteModel
from keras.optimizers import Adam
from sample_validation import validation_load


def main():
    outputpath = "output/skipgram_discrete"
    ds = load_dataset()
    vd = validation_load()
    batch_size = 256
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 10
    units = 512
    embedding_units = 128
    z_k = 2
    z_depth = 5
    # kernel_regularizer = L1L2(1e-9, 1e-9)
    # embeddings_regularizer = L1L2(1e-9, 1e-9)
    # embeddings_regularizer = None
    loss_weight = 1e-2
    opt = Adam(1e-4)
    opt_a = Adam(1e-3)
    adversary_weight = 0
    layernorm = False
    batchnorm = True
    model = SkipgramDiscreteModel(dataset=ds,
                                  z_k=z_k,
                                  z_depth=z_depth,
                                  window=window,
                                  embedding_units=embedding_units,
                                  adversary_weight=adversary_weight,
                                  loss_weight=loss_weight,
                                  opt=opt,
                                  opt_a=opt_a,
                                  layernorm=layernorm,
                                  batchnorm=batchnorm,
                                  inner_activation=leaky_relu,
                                  units=units)
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
