# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

from dataset_util import load_dataset
from discrete_skip_gram.layers.utils import leaky_relu
from discrete_skip_gram.one_bit_models.one_bit_discrete_model import OneBitDiscreteModel


def main():
    outputpath = "output/one_bit_discrete"
    ds = load_dataset()
    batch_size = 256
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
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                output_path=outputpath,
                frequency=frequency)


if __name__ == "__main__":
    main()
