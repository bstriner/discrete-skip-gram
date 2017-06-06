"""
Each element of sequence is an embedding layer
"""
import csv
import os

import numpy as np
from keras.layers import Input, Lambda, Dense, Activation, BatchNormalization, Flatten
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T

from discrete_skip_gram.layers.utils import leaky_relu
from ..datasets.mnist_dataset import mnist_data
from ..layers.one_bit_bayes_layer import OneBitBayesLayer
from ..layers.one_bit_validation_layer import OneBitValidationLayer
from ..layers.uniform_smoothing import UniformSmoothing
from ..layers.utils import softmax_nd_layer
from .mnist_model import MnistModel


class OneBitBayesModel(MnistModel):
    """
    MNIST one-bit bayesian model
    """
    def __init__(self,
                 z_k,
                 lr=1e-4,
                 units=512,
                 inner_activation=leaky_relu,
                 hidden_layers=2):
        super(OneBitBayesModel, self).__init__()
        self.hidden_layers = hidden_layers
        self.z_k = z_k
        self.inner_activation = inner_activation

        input_x = Input((28, 28,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        # p(z|x)
        h = Lambda(lambda _x: T.cast(_x, 'float32'), output_shape=lambda _x:_x)(input_x)
        h = Flatten()(h)
        h = BatchNormalization()(h)
        for i in range(3):
            h = Dense(units)(h)
            h = BatchNormalization()(h)
            h = Activation(inner_activation)(h)
        h = Dense(z_k)(h)
        h= softmax_nd_layer()(h)
        p_z_given_x = UniformSmoothing()(h)

        # nll
        layer = OneBitBayesLayer(y_k=10, z_k=z_k)
        prior_nll, posterior_nll = layer([p_z_given_x, input_y])

        # validation nll
        val_layer = OneBitValidationLayer(y_k=10, z_k=z_k)
        val_nll = val_layer([p_z_given_x, input_y])  # (n,1)

        tot_loss = Lambda(lambda (_a, _b, _c): _a + _b + _c,
                          output_shape=lambda (_a, _b, _c): _a)(
            [val_nll, prior_nll, posterior_nll])

        def avg_prior_nll(ytrue, ypred):
            return T.mean(prior_nll, axis=None)

        def avg_posterior_nll(ytrue, ypred):
            return T.mean(posterior_nll, axis=None)

        def avg_val_nll(ytrue, ypred):
            return T.mean(val_nll, axis=None)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[tot_loss])
        self.model.compile(opt, loss_f, metrics=[avg_prior_nll, avg_posterior_nll, avg_val_nll])
        self.weights = self.model.weights + opt.weights

        # Encoder model
        z_sampled = Lambda(lambda _z: T.argmax(_z, axis=1, keepdims=True),
                           output_shape=lambda _z: (_z[0], 1))(p_z_given_x)
        self.model_encode = Model([input_x], z_sampled)

    def on_epoch_end(self, output_path, epoch):
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
