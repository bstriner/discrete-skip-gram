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


class SkipgramValidationModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 embedding,
                 units,
                 embedding_units,
                 window,
                 z_k,
                 lr=1e-4,
                 lr_a=1e-3,
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

        x_embedding = SequentialEmbeddingDiscrete(embedding)
        z = drop_dim_2()(x_embedding(input_x))

        zrnn = HighwayLayerDiscrete(units=self.units,
                                    embedding_units=embedding_units,
                                    k=z_k,
                                    layernorm=layernorm,
                                    inner_activation=self.inner_activation,
                                    hidden_layers=self.hidden_layers,
                                    kernel_regularizer=kernel_regularizer)
        print "Z dim: {}".format(z.ndim)
        zh = zrnn(z)  # n, z_depth, units

        h = zh
        for i in range(3):
            h = TimeDistributedDense(units=self.units,
                                     activation=self.inner_activation,
                                     kernel_regularizer=kernel_regularizer)(h)
        p = TimeDistributedDense(units=x_k,
                                 kernel_regularizer=kernel_regularizer,
                                 activation=softmax_nd)(h)  # (n, z_depth, x_k)

        eps = 1e-7
        scale = 1 - (eps * x_k)
        nll = Lambda(lambda (_p, _y): -T.log(eps + (scale * _p[T.arange(_p.shape[0]), :, T.flatten(_y)])),
                     output_shape=lambda (_p, _y): (_p[0], _p[1]))([p, input_y])  # (n, z_depth)

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

        # Prediction model
        pfinal = Lambda(lambda _p: _p[:, -1, :],
                        output_shape=lambda _p: (_p[0], _p[2]))(p)  # n, x_k
        policy_sampler = SamplerLayer(srng=srng)
        ygen = policy_sampler(pfinal)
        self.model_predict = Model([input_x], ygen)

    def on_epoch_end(self, output_path, epoch):
        self.write_predictions("{}/predictions-{:08d}.csv".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
