import os
os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"

from dataset_util import load_dataset
from discrete_skip_gram.one_bit_models.one_bit_em_model import OneBitEMModel


def main():
    outputpath = "output/one_bit_em"
    ds = load_dataset()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 5
    lr = 3e-4
    model = OneBitEMModel(dataset=ds,
                          window=window,
                          lr=lr)
    model.summary()
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                output_path=outputpath,
                frequency=frequency)


if __name__ == "__main__":
    main()
