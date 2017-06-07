"""
Each element of sequence is an embedding layer
"""
import numpy as np
from theano import tensor as T

import keras.backend as K
from discrete_skip_gram.layers.utils import leaky_relu
from keras.layers import Input, Embedding, Lambda
from keras.models import Model
from keras.optimizers import Adam
from ..layers.sequential_embedding_discrete import SequentialEmbeddingDiscrete
from ..layers.uniform_smoothing import UniformSmoothing
from ..layers.utils import nll_metrics
from ..layers.utils import softmax_nd_layer, drop_dim_2
from ..skipgram_models.skipgram_model import SkipgramModel


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
                 embeddings_regularizer=None,
                 hidden_layers=2,
                 ):
        self.dataset = dataset
        self.units = units
        self.embedding_units = embedding_units
        self.window = window
        self.hidden_layers = hidden_layers
        self.z_depth = embedding.shape[1]
        self.z_k = z_k
        self.inner_activation = inner_activation
        x_k = self.dataset.k
        assert x_k == embedding.shape[0]

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        x_embedding = SequentialEmbeddingDiscrete(embedding)  # n, 1, 1
        z = drop_dim_2()(x_embedding(input_x))  # n, 1

        h = Embedding(input_dim=z_k,
                      embeddings_regularizer=embeddings_regularizer,
                      output_dim=x_k)(z)  # n, 1, x_k
        h = drop_dim_2()(h)
        h = softmax_nd_layer()(h)
        p = UniformSmoothing()(h)  # n, x_k

        nll = Lambda(lambda (_p, _y): T.reshape(-T.log(_p[T.arange(_p.shape[0]), T.flatten(_y)]),
                                                (-1, 1)),
                     output_shape=lambda (_p, _y): (_p[0], 1))([p, input_y])  # (n, 1)

        self.loss_weight = K.variable(np.float32(loss_weight), dtype='float32', name='loss_weight')

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None) * self.loss_weight

        avg_nll = T.mean(nll, axis=0)
        metrics = nll_metrics(avg_nll, self.z_depth)
        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[nll])
        self.model.compile(opt, loss_f, metrics=metrics)
        self.weights = self.model.weights + opt.weights

    def on_epoch_end(self, output_path, epoch):
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
