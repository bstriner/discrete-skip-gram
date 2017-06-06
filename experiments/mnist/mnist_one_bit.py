# import os
# os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"
from discrete_skip_gram.layers.utils import leaky_relu
from discrete_skip_gram.mnist_models.one_bit_model import OneBitModel


def experiment(outputpath, z_k):
    batch_size = 256
    epochs = 20
    steps_per_epoch = 2048
    frequency = 5
    lr = 1e-3
    model = OneBitModel(z_k=z_k,
                        inner_activation=leaky_relu,
                        lr=lr)
    model.summary()

    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                output_path=outputpath,
                frequency=frequency)


def main():
    experiment("output/mnist/one_bit_2", 2)
    experiment("output/mnist/one_bit_5", 5)


if __name__ == "__main__":
    main()
