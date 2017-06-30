"""
Each element of sequence is an embedding layer
"""
import numpy as np
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import keras.backend as K
from discrete_skip_gram.layers.utils import leaky_relu
from keras.layers import Input, Lambda, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import L1L2
from .skipgram_model import SkipgramModel
from ..dataset_util import load_dataset
from ..layers.highway_layer_discrete import HighwayLayerDiscrete
from ..layers.sequential_embedding_discrete import SequentialEmbeddingDiscrete
from ..layers.time_distributed_dense import TimeDistributedDense
from ..layers.unrolled.sampler_layer import SamplerLayer
from ..layers.utils import nll_metrics
from ..layers.utils import softmax_nd, drop_dim_2


class SkipgramValidationModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 embedding,
                 units,
                 embedding_units,
                 window,
                 schedule,
                 z_k,
                 lr=1e-4,
                 loss_weight=1e-2,
                 inner_activation=leaky_relu,
                 kernel_regularizer=None,
                 hidden_layers=2,
                 layernorm=False,
                 batchnorm=True
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
        assert len(schedule.shape) == 1
        assert schedule.shape[0] == self.z_depth

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        x_embedding = SequentialEmbeddingDiscrete(embedding)
        z = drop_dim_2()(x_embedding(input_x))

        zrnn = HighwayLayerDiscrete(units=self.units,
                                    embedding_units=embedding_units,
                                    k=z_k,
                                    layernorm=layernorm,
                                    batchnorm=batchnorm,
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
            if batchnorm:
                h = BatchNormalization()(h)
        p = TimeDistributedDense(units=x_k,
                                 kernel_regularizer=kernel_regularizer,
                                 activation=softmax_nd)(h)  # (n, z_depth, x_k)

        eps = 1e-7
        scale = 1 - (eps * x_k)
        nll = Lambda(lambda (_p, _y): -T.log(eps + (scale * _p[T.arange(_p.shape[0]), :, T.flatten(_y)])),
                     output_shape=lambda (_p, _y): (_p[0], _p[1]))([p, input_y])  # (n, z_depth)

        schedule_var = T.constant(schedule, dtype='float32', name='schedule')
        loss = Lambda(lambda _nll: T.sum(_nll * (schedule_var.dimshuffle(('x', 0))), axis=1, keepdims=True),
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
        self.model._make_train_function()
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


def validate_skipgram(outputpath, embeddingpath):
    embedding = np.load(embeddingpath)
    ds = load_dataset()
    batch_size = 256
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 10
    units = 512
    embedding_units = 128
    z_k = 2
    kernel_regularizer = L1L2(1e-9, 1e-9)
    loss_weight = 1e-2
    lr = 3e-4
    layernorm = False
    batchnorm = True
    z_depth = embedding.shape[1]
    schedule = np.power(1.5, np.arange(z_depth))
    model = SkipgramValidationModel(dataset=ds,
                                    z_k=z_k,
                                    schedule=schedule,
                                    embedding=embedding,
                                    window=window,
                                    embedding_units=embedding_units,
                                    kernel_regularizer=kernel_regularizer,
                                    loss_weight=loss_weight,
                                    layernorm=layernorm,
                                    batchnorm=batchnorm,
                                    inner_activation=leaky_relu,
                                    units=units,
                                    lr=lr)
    model.summary()
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                output_path=outputpath,
                frequency=frequency)
