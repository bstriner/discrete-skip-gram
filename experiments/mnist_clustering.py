# import os
# os.environ["THEANO_FLAGS"]='optimizer=None,device=cpu'
import numpy as np
import theano.tensor as T
from keras.datasets import mnist
from keras.initializers import RandomUniform
from keras.optimizers import Adam

from discrete_skip_gram.fcgan import FCGAN
from discrete_skip_gram.mlp import MLP
from discrete_skip_gram.regularizers import BalanceRegularizer
from discrete_skip_gram.tensor_util import leaky_relu, softmax_nd


def main():
    units = 512
    rng_units = 128
    z_k = 10
    pz_regularizer = BalanceRegularizer(1e-2)
    iwgan_weight = 1e-1

    initializer = RandomUniform(minval=-0.05, maxval=0.05)
    ((x, y), (xt, yt)) = mnist.load_data()
    x = np.float32(x) / 255.
    x = np.reshape(x, (x.shape[0], -1))
    input_units = 28 * 28
    classifier = MLP(input_units=input_units,
                     hidden_units=units,
                     output_units=z_k,
                     hidden_depth=2,
                     hidden_activation=leaky_relu,
                     initializer=initializer,
                     output_activation=softmax_nd)
    generator = MLP(input_units=units,
                    hidden_units=units,
                    output_units=input_units,
                    hidden_depth=2,
                    hidden_activation=leaky_relu,
                    initializer=initializer,
                    output_activation=T.nnet.sigmoid)
    discriminator = MLP(input_units=units,
                        hidden_units=units,
                        output_units=1,
                        hidden_depth=2,
                        initializer=initializer,
                        hidden_activation=leaky_relu)
    model = FCGAN(
        z_k=z_k,
        classifier=classifier,
        generator=generator,
        discriminator=discriminator,
        optd=Adam(1e-3),
        optg=Adam(1e-3),
        input_units=input_units,
        rng_units=rng_units,
        units=units,
        activation=leaky_relu,
        pz_regularizer=pz_regularizer,
        initializer=initializer,
        iwgan_weight=iwgan_weight
    )
    model.train(
        x=x,
        output_path='output/mnist_clustering',
        epochs=500,
        batches=512,
        discriminator_batches=5,
        batch_size=128
    )


if __name__ == '__main__':
    main()
