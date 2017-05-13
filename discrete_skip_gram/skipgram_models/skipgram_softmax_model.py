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
from ..layers.utils import softmax_nd_layer, shift_tensor_layer, softmax_nd
from ..layers.highway_layer_discrete import HighwayLayerDiscrete
from .skipgram_model import SkipgramModel


class SkipgramSoftmaxModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 units,
                 embedding_units,
                 window,
                 z_depth,
                 z_k,
                 lr=1e-4,
                 lr_a=1e-3,
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
        self.z_depth = z_depth
        self.z_k = z_k
        self.inner_activation = inner_activation
        assert z_depth > 0
        srng = RandomStreams(123)
        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        x_embedding = Embedding(input_dim=self.dataset.k,
                                output_dim=z_depth * z_k,
                                embeddings_regularizer=embeddings_regularizer)
        rs = Reshape((z_depth, z_k))
        sm = softmax_nd_layer()
        z = sm(rs(x_embedding(input_x)))  # n, z_depth, z_k

        # sampler = Sampler3DLayer(srng=srng)
        sampler = Lambda(lambda _x: T.argmax(_x, axis=2),
                         output_shape=lambda _x: (_x[0], _x[1]),
                         name='z_sampler')
        z_sampled = sampler(z)
        z_shifted = shift_tensor_layer()(z_sampled)

        zrnn = HighwayLayerDiscrete(units=self.units,
                                    embedding_units=self.embedding_units,
                                    k=self.z_k + 1,
                                    layernorm=layernorm,
                                    inner_activation=self.inner_activation,
                                    hidden_layers=self.hidden_layers,
                                    kernel_regularizer=kernel_regularizer)

        zh = zrnn(z_shifted)  # n, z_depth, units

        h = zh
        for i in range(3):
            h = TimeDistributedDense(units=self.units,
                                     activation=self.inner_activation,
                                     kernel_regularizer=kernel_regularizer)(h)
        h = TimeDistributedDense(units=self.z_k * x_k,
                                 kernel_regularizer=kernel_regularizer)(h)
        h = Reshape((z_depth, self.z_k, x_k))(h)  # (n, z_depth, z_k, x_k)
        p = softmax_nd_layer()(h)  # (n, z_depth, z_k, x_k)
        eps = 1e-7
        nll = Lambda(lambda (_p, _y): -T.log(eps+_p[T.arange(_p.shape[0]), :, :, T.flatten(_y)]),
                     output_shape=lambda (_p, _y): (_p[0], _p[1], _p[2]))([p, input_y])

        loss = Lambda(lambda (_nll, _z): T.sum(T.sum(_nll * _z, axis=2), axis=1, keepdims=True),
                      output_shape=lambda (_nll, _z): (_nll[0], 1),
                      name="loss_layer")([nll, z])

        # adversary: try to minimize kl with z
        input_z0 = Input((self.z_depth,), dtype='int32', name='input_z0')
        arnn = HighwayLayerDiscrete(units=units,
                                    embedding_units=embedding_units,
                                    k=self.z_k + 1,
                                    layernorm=layernorm,
                                    inner_activation=inner_activation,
                                    hidden_layers=hidden_layers,
                                    kernel_regularizer=kernel_regularizer)
        ah = arnn(input_z0)
        h = ah
        for i in range(3):
            h = TimeDistributedDense(units=self.units,
                                     kernel_regularizer=kernel_regularizer,
                                     activation=self.inner_activation)(h)
        d = TimeDistributedDense(units=self.z_k,
                                 activation=softmax_nd,
                                 kernel_regularizer=kernel_regularizer)(h)
        adversary = Model(inputs=[input_z0], outputs=[d])

        # train adversary
        dz = adversary(z_shifted)

        def em(yt, yp):
            return T.mean(T.sum(T.sum(T.abs_(yt - yp), axis=2), axis=1), axis=0)

        aem = em(z, dz)
        adversary_reg = 0
        for layer in adversary.layers:
            for l in layer.losses:
                adversary_reg += l
        aopt = Adam(lr=lr_a)
        atotal = aem + adversary_reg
        aweights = adversary.trainable_weights
        print "A weights: {}".format(aweights)
        aupdates = aopt.get_updates(aweights, {}, atotal)
        self.adversary_weight = K.variable(np.float32(adversary_weight), dtype='float32', name='adversary_weight')
        regloss = -self.adversary_weight * aem

        self.adversary = adversary
        zrnn.add_loss(regloss)
        zrnn.add_update(aupdates)

        self.model = Model([input_x, input_y], loss)

        def adversary_em(ytrue, ypred):
            return aem

        def adversary_loss(ytrue, ypred):
            return atotal

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        avg_nll = T.mean(nll, axis=0)
        metrics = nll_metrics(avg_nll, z_depth)
        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[loss])
        self.model.compile(opt, loss_f, metrics=[adversary_loss, adversary_em] + metrics)
        self.weights = self.model.weights + self.adversary.weights + opt.weights + aopt.weights

        # Prediction model
        # p: (n, z_depth, z_k, x_k)
        # z_sampled: (n, z_depth) [int 0-z_k]
        pfinal = Lambda(lambda (_p, _z_sampled): _p[T.arange(p.shape[0]), -1, T.flatten(_z_sampled[:,-1]), :],
                        output_shape=lambda (_p, _z_sampled): (_p[0], x_k))([p, z_sampled]) # n, x_k
        policy_sampler = SamplerLayer(srng=srng)
        ygen = policy_sampler(pfinal)
        self.model_predict = Model([input_x], ygen)

        # Encoder model
        self.model_encode = Model([input_x], z_sampled)

        idx0 = T.extra_ops.repeat(T.reshape(T.arange(p.shape[0]), (-1,1)), p.shape[1], axis=1)
        idx1 = T.extra_ops.repeat(T.reshape(T.arange(p.shape[1]), (1,-1)), p.shape[0], axis=0)
        prob = Lambda(lambda (_p, _z): _p[idx0, idx1, _z, :],
                      output_shape=lambda (_p, _z): (_p[0], _p[1], _p[3]))([p, z_sampled])
        # prob: (n, z_depth, x_k)
        self.model_probability = Model([input_x], prob)

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

    def on_epoch_end(self, output_path, epoch):
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.write_predictions("{}/predictions-{:08d}.csv".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))