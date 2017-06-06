# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np

from dataset_util import load_dataset
from discrete_skip_gram.layers.utils import leaky_relu
from discrete_skip_gram.one_bit_models.one_bit_discrete_model import OneBitDiscreteModel
from sample_validation import validation_load


def main():
    outputpath = "output/one_bit_discrete"
    ds = load_dataset()
    vd = validation_load()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 10
    lr = 3e-4
    z_k = 2
    model = OneBitDiscreteModel(dataset=ds,
                                z_k=z_k,
                                window=window,
                                inner_activation=leaky_relu,
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
