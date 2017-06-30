# import os
# os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"

import numpy as np

from discrete_skip_gram.dataset_util import load_dataset
from discrete_skip_gram.layers.utils import leaky_relu
from discrete_skip_gram.one_bit_models.one_bit_validation_model import OneBitValidationModel
from sample_validation import validation_load


def main():
    outputpath = "output/one_bit_random"
    z_k = 2
    batch_size = 256
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 20
    units = 512
    embedding_units = 128
    loss_weight = 1e-2
    lr = 3e-4
    ds = load_dataset()
    vd = validation_load()
    x_k = ds.k
    embedding = np.random.randint(0, z_k, (x_k, 1))
    print "K: {}, {}, {}".format(z_k, x_k, np.max(embedding))
    model = OneBitValidationModel(dataset=ds,
                                  z_k=z_k,
                                  embedding=embedding,
                                  window=window,
                                  embedding_units=embedding_units,
                                  loss_weight=loss_weight,
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
