"""
Each element of sequence is an embedding layer
"""
import os
import numpy as np
from keras.layers import Input, Embedding, Lambda, Dense
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from ..layers.unrolled.sampler_layer import SamplerLayer
from ..layers.utils import drop_dim_2
from discrete_skip_gram.layers.utils import leaky_relu
from .skipgram_model import SkipgramModel
from ..layers.utils import zeros_layer, softmax_nd_layer
from ..layers.unrolled.bias_layer import BiasLayer
class UnigramBaselineModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 units,
                 window,
                 lr=1e-4,
                 inner_activation=leaky_relu,
                 kernel_regularizer=None,
                 hidden_layers=2
                 ):
        self.dataset = dataset
        self.units = units
        self.window = window
        self.hidden_layers = hidden_layers
        self.inner_activation = inner_activation

        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')


        zb = BiasLayer(x_k, bias_regularizer=kernel_regularizer)
        z = zb(input_x)
        p = softmax_nd_layer()(z)
        eps = 1e-7
        nll = Lambda(lambda (_p, _y): -T.log(eps + _p[T.arange(_p.shape[0]), T.flatten(_y)]),
                     output_shape=lambda (_p, _y): (_p[0], 1))([p, input_y])

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def avg_nll(ytrue, ypred):
            return T.mean(nll, axis=None)

        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[nll])
        self.model.compile(opt, loss_f, metrics=[avg_nll])
        self.weights = self.model.weights + opt.weights

        # prob: (n, z_depth, x_k)
        self.model_probability = Model([input_x], p)

    def on_epoch_end(self, output_path, epoch):
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
