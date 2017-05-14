"""
Each element of sequence is an embedding layer
"""
import keras.backend as K
import csv
import os
import numpy as np
from keras.layers import Input, Embedding, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from ..layers.unrolled.sampler_layer import SamplerLayer
from ..layers.time_distributed_dense import TimeDistributedDense
from ..layers.utils import nll_metrics

from discrete_skip_gram.layers.utils import leaky_relu
from ..layers.utils import softmax_nd_layer, softmax_nd, drop_dim_2
from .skipgram_model import SkipgramModel
from ..layers.highway_layer_discrete import HighwayLayerDiscrete
from ..layers.highway_layer import HighwayLayer
from ..layers.shift_padding_layer import ShiftPaddingLayer
from ..layers.sequential_embedding_discrete import SequentialEmbeddingDiscrete


class OneBitValidationModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 embedding,
                 units,
                 embedding_units,
                 window,
                 z_k,
                 lr=1e-4,
                 loss_weight=1e-2,
                 inner_activation=leaky_relu,
                 kernel_regularizer=None,
                 embeddings_regularizer=None,
                 hidden_layers=2,
                 layernorm=False,
                 adversary_weight=1.0
                 ):
        self.dataset = dataset
        self.units = units
        self.embedding_units = embedding_units
        self.window = window
        self.hidden_layers = hidden_layers
        self.z_depth = embedding.shape[1]
        self.z_k = z_k
        self.inner_activation = inner_activation
        srng = RandomStreams(123)
        x_k = self.dataset.k
        assert x_k == embedding.shape[0]

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        x_embedding = SequentialEmbeddingDiscrete(embedding)  # n, 1, 1
        z = drop_dim_2()(x_embedding(input_x))  # n, 1

        h = Embedding(input_dim=z_k,
                      embeddings_regularizer=embeddings_regularizer,
                      output_dim=x_k)(z)  # n, 1, x_k
        p = softmax_nd_layer()(drop_dim_2()(h))  # n, x_k

        eps = 1e-7
        scale = 1.0 - (eps * x_k)
        nll = Lambda(lambda (_p, _y): T.reshape(-T.log(eps + (scale * _p[T.arange(_p.shape[0]), T.flatten(_y)])),
                                                (-1, 1)),
                     output_shape=lambda (_p, _y): (_p[0], 1))([p, input_y])  # (n, z_depth)

        loss = Lambda(lambda _nll: T.sum(_nll, axis=1, keepdims=True),
                      output_shape=lambda _nll: (_nll[0], 1),
                      name="loss_layer")(nll)  # (n, 1)

        self.loss_weight = K.variable(np.float32(loss_weight), dtype='float32', name='loss_weight')
        self.model = Model([input_x, input_y], loss)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None) * self.loss_weight

        avg_nll = T.mean(nll, axis=0)
        metrics = nll_metrics(avg_nll, self.z_depth)
        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[loss])
        self.model.compile(opt, loss_f, metrics=metrics)
        self.weights = self.model.weights + opt.weights

    def on_epoch_end(self, output_path, epoch):
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
