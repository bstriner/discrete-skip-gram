"""
Each element of sequence is an embedding layer
"""
import csv
import os

import numpy as np
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import keras.backend as K
from discrete_skip_gram.layers.utils import leaky_relu
from keras.layers import BatchNormalization
from keras.layers import Input, Lambda
from keras.models import Model
from .skipgram_model import SkipgramModel
from ..layers.dense_batch import DenseBatch
from ..layers.highway_layer import HighwayLayer
from ..layers.nll_layer import NLL
from ..layers.prior_layer import PriorLayer
from ..layers.sequential_embedding_balanced_binary import SequentialEmbeddingBalancedBinary
from ..layers.sequential_embedding_simple_binary import SequentialEmbeddingSimpleBinary
from ..layers.shift_padding_layer import ShiftPaddingLayer
from ..layers.time_distributed_dense import TimeDistributedDense
from ..layers.uniform_smoothing import UniformSmoothing
from ..layers.utils import nll_metrics
from ..layers.utils import softmax_nd_layer


class SkipgramDiscreteBinaryModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 units,
                 embedding_units,
                 window,
                 z_depth,
                 opt,
                 inner_activation=leaky_relu,
                 kernel_regularizer=None,
                 embeddings_regularizer=None,
                 hidden_layers=2,
                 layernorm=False,
                 batchnorm=True,
                 loss_weight=1e-2,
                 balancer=True,
                 dense_batch=True,
                 do_prior=False,
                 do_validate=False
                 ):
        if dense_batch:
            dense_cls = DenseBatch
        else:
            dense_cls = TimeDistributedDense
        self.dataset = dataset
        self.units = units
        self.embedding_units = embedding_units
        self.window = window
        self.hidden_layers = hidden_layers
        self.z_depth = z_depth
        self.inner_activation = inner_activation
        assert z_depth > 0
        srng = RandomStreams(123)
        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        losses = []
        metrics = []

        # p(y) prior
        if do_prior:
            prior_layer = PriorLayer(x_k=x_k)
            p_y = prior_layer(input_y)
            prior_nll = NLL()(p_y)
            losses.append(prior_nll)

            def avg_prior_nll(ytrue, ypred):
                return T.mean(prior_nll, axis=None)

            metrics.append(avg_prior_nll())

        # p(z|x)
        if balancer:
            x_embedding = SequentialEmbeddingBalancedBinary(x_k=x_k,
                                                            z_depth=z_depth,
                                                            units=self.units,
                                                            batchnorm=batchnorm,
                                                            srng=srng,
                                                            inner_activation=self.inner_activation)
        else:
            x_embedding = SequentialEmbeddingSimpleBinary(x_k=x_k, z_depth=z_depth)
        p_z_given_x, z_sampled = x_embedding(input_x)
        # p_z_given_x (n, z_depth)
        # z_sampled (n, z_depth)
        z_sampled_3d = Lambda(lambda _x: _x.dimshuffle((0, 1, 'x')),
                              output_shape=lambda _x: (_x[0], _x[1], 1))(z_sampled)  # (n, z_depth, 1)

        # p(y|z)
        zrnn = HighwayLayer(units=self.units,
                            layernorm=layernorm,
                            inner_activation=self.inner_activation,
                            hidden_layers=self.hidden_layers,
                            batchnorm=batchnorm,
                            kernel_regularizer=kernel_regularizer)
        h = zrnn(z_sampled_3d)  # n, z_depth, units
        h = ShiftPaddingLayer()(h)
        for i in range(2):
            h = dense_cls(units=self.units,
                          activation=self.inner_activation,
                          kernel_regularizer=kernel_regularizer)(h)
            if batchnorm:
                h = BatchNormalization()(h)  # (n, z_depth, units)
        p_bias = dense_cls(units=x_k,
                           kernel_regularizer=kernel_regularizer)(h)  # (n, z_depth, x_k)
        p_weight = dense_cls(units=x_k,
                             kernel_regularizer=kernel_regularizer)(h)  # (n, z_depth, x_k)
        h = Lambda(lambda (_b, _w, _p): _b + (_p.dimshuffle((0, 1, 'x')) * _w),
                   output_shape=lambda (_b, _w, _p): _b)([p_bias, p_weight, p_z_given_x])  # (n, z_depth, y_k)
        h = softmax_nd_layer()(h)
        p_y_given_z = UniformSmoothing()(h)  # (n, z_depth, y_k)
        p_y_given_z_t = Lambda(lambda (_p, _y): _p[T.arange(_p.shape[0]), :, T.flatten(_y)],
                               output_shape=lambda (_p, _y): (_p[0], _p[1],))(
            [p_y_given_z, input_y])  # (n, z_depth)
        nll = NLL()(p_y_given_z_t)
        loss = Lambda(lambda _nll: T.sum(nll, axis=1, keepdims=True),
                      output_shape=lambda _nll: (_nll[0], 1))(nll)  # (n, 1)
        losses.append(loss)
        avg_nll = T.mean(nll, axis=0)
        metrics += nll_metrics(avg_nll, z_depth)

        if do_validate:
            # validation model
            zrnnval = HighwayLayer(units=self.units,
                                   layernorm=layernorm,
                                   inner_activation=self.inner_activation,
                                   hidden_layers=self.hidden_layers,
                                   batchnorm=batchnorm,
                                   kernel_regularizer=kernel_regularizer)
            h = zrnnval(z_sampled_3d)  # n, z_depth, units
            for i in range(2):
                h = dense_cls(units=self.units,
                              activation=self.inner_activation,
                              kernel_regularizer=kernel_regularizer)(h)
                if batchnorm:
                    h = BatchNormalization()(h)  # (n, z_depth, units)
            h = dense_cls(units=x_k,
                          kernel_regularizer=kernel_regularizer)(h)  # (n, z_depth, x_k)
            h = softmax_nd_layer()(h)
            p_val = UniformSmoothing()(h)  # (n, z_depth, y_k)
            p_val_t = Lambda(lambda (_p, _y): _p[T.arange(_p.shape[0]), :, T.flatten(_y)],
                             output_shape=lambda (_p, _y): (_p[0], _p[1],))(
                [p_val, input_y])  # (n, z_depth)
            val_nll = NLL()(p_val_t)
            val_loss = Lambda(lambda _x: T.sum(_x, axis=1, keepdims=True),
                              output_shape=lambda _x: (_x[0], 1))(val_nll)
            avg_val_nll = T.mean(val_nll, axis=0)
            losses.append(val_loss)
            metrics += nll_metrics(avg_val_nll, z_depth, fmt="val_nll{:02d}")

        # total_loss = Lambda(lambda (_a, _b, _c): _a + _b + _c,
        #                    output_shape=lambda (_a, _b, _c): _a)([loss, prior_nll, val_loss])
        if len(losses) == 1:
            total_loss = losses[0]
        else:
            total_loss = Lambda(lambda _x: sum(_x),
                                output_shape=lambda _x: _x[0])(losses)
        self.loss_weight = K.variable(np.float32(loss_weight), dtype='float32', name='loss_weight')

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None) * self.loss_weight

        self.model = Model(inputs=[input_x, input_y], outputs=[total_loss])
        self.model.compile(opt, loss_f, metrics=metrics)
        self.weights = self.model.weights + opt.weights

        # Encoder model
        self.model_encode = Model([input_x], z_sampled)
        # z model
        self.model_z = Model([input_x], p_z_given_x)

    def write_encodings(self, output_path):
        x = np.arange(self.dataset.k).reshape((-1, 1))
        z = self.model_encode.predict(x, verbose=0)

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path + ".csv", 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word", "Encoding"] + ["Cat {}".format(j) for j in range(self.z_depth)])
            for i in range(self.dataset.k):
                enc = z[i, :]
                enca = [enc[j] for j in range(enc.shape[0])]
                encf = "".join(chr(ord('a') + e) for e in enca)
                word = self.dataset.get_word(i)
                w.writerow([i, word, encf] + enca)
        np.save(output_path + ".npy", z)

    def write_z(self, output_path):
        x = np.arange(self.dataset.k).reshape((-1, 1))
        z = self.model_z.predict(x, verbose=0)
        np.save(output_path + ".npy", z)

    def on_epoch_end(self, output_path, epoch):
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.write_z("{}/z-{:08d}".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
