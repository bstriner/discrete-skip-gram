# import os
# os.environ["THEANO_FLAGS"]='optimizer=None,device=cpu'
import numpy as np
import theano.tensor as T
from keras.datasets import mnist
from keras.initializers import RandomUniform
from keras.optimizers import Adam

from discrete_skip_gram.mlp import MLP
from discrete_skip_gram.mse_model import MSEModel
from discrete_skip_gram.regularizers import BalanceRegularizer
from discrete_skip_gram.tensor_util import leaky_relu, softmax_nd


def main():
    units = 512
    encoding_units = 128
    z_k = 10
    input_units = 28 * 28
    pz_regularizer = BalanceRegularizer(1e-2)
    reg_weight_encoding = 1e-3
    reg_weight_grad = 1e-3

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
    encoder = MLP(input_units=input_units,
                  hidden_units=units,
                  output_units=encoding_units,
                  hidden_depth=2,
                  hidden_activation=leaky_relu,
                  initializer=initializer,
                  output_activation=leaky_relu)
    generator = MLP(input_units=units,
                    hidden_units=units,
                    output_units=input_units,
                    hidden_depth=2,
                    hidden_activation=leaky_relu,
                    initializer=initializer,
                    output_activation=T.nnet.sigmoid)

    model = MSEModel(
        z_k=z_k,
        classifier=classifier,
        encoder=encoder,
        generator=generator,
        opt=Adam(1e-3),
        encoding_units=encoding_units,
        units=units,
        activation=leaky_relu,
        reg_weight_encoding=reg_weight_encoding,
        reg_weight_grad=reg_weight_grad,
        initializer=initializer,
        pz_regularizer=pz_regularizer
    )
    model.train(
        x=x,
        output_path='output/mnist_clustering_mse',
        epochs=500,
        batches=512,
        batch_size=128
    )


if __name__ == '__main__':
    main()
