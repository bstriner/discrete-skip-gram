# import os
# os.environ["THEANO_FLAGS"]='optimizer=None,device=cpu'

import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
import numpy as np
from keras.datasets import mnist
from keras.initializers import RandomUniform
from keras.optimizers import Adam

from discrete_skip_gram.mlp import MLP
from discrete_skip_gram.mse_model import MSEModel
from discrete_skip_gram.regularizers import BalanceRegularizer
from discrete_skip_gram.tensor_util import leaky_relu, softmax_nd
from keras.regularizers import L1L2
import theano.tensor as T


def main():
    units = 512
    encoding_units = 128
    z_k = 10
    input_units = 28 * 28
    pz_regularizer = BalanceRegularizer(1e-1)
    reg_weight_encoding = 1e-1
    gen_regularizer = L1L2(1e-2, 1e-2)

    ((x, y), (xt, yt)) = mnist.load_data()
    x = np.float32(x) / 255.
    x = np.reshape(x, (x.shape[0], -1))
    initializer = RandomUniform(minval=-0.05, maxval=0.05)
    classifier = MLP(input_units=input_units,
                     hidden_units=units,
                     output_units=z_k,
                     hidden_depth=2,
                     hidden_activation=leaky_relu,
                     initializer=initializer,
                     output_activation=softmax_nd)
    encoder = MLP(input_units=units,
                  hidden_units=units,
                  output_units=encoding_units,
                  hidden_depth=2,
                  hidden_activation=leaky_relu,
                  initializer=initializer,
                  output_activation=leaky_relu)
    generator = MLP(input_units=encoding_units,
                    hidden_units=units,
                    output_units=input_units,
                    hidden_depth=2,
                    hidden_activation=leaky_relu,
                    initializer=initializer,
                    output_activation=T.nnet.sigmoid)

    model = MSEModel(
        z_k=z_k,
        mode=2,
        classifier=classifier,
        encoder=encoder,
        generator=generator,
        units=units,
        encoding_units=encoding_units,
        opt=Adam(1e-3),
        input_units=input_units,
        reg_weight_encoding=reg_weight_encoding,
        activation=leaky_relu,
        initializer=initializer,
        pz_regularizer=pz_regularizer,
        gen_regularizer=gen_regularizer
    )
    model.train(
        x=x,
        output_path='output/mnist_clustering_mse',
        epochs=500,
        batches=10000,
        batch_size=128
    )


if __name__ == '__main__':
    main()
