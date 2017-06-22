"""
Each element of sequence is an embedding layer
"""
import numpy as np
from theano import tensor as T

import keras.backend as K
from discrete_skip_gram.layers.utils import leaky_relu
from keras.layers import Input, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from .skipgram_model import SkipgramModel
from ..layers.sequential_embedding_discrete import SequentialEmbeddingDiscrete
from ..layers.skipgram_validation_flat_layer import SkipgramValidationFlatLayer
from ..layers.utils import nll_metrics


class SkipgramValidationFlatModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 embedding,
                 units,
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
        self.window = window
        self.hidden_layers = hidden_layers
        self.z_depth = embedding.shape[1]
        self.z_k = z_k
        self.inner_activation = inner_activation
        x_k = self.dataset.k
        assert x_k == embedding.shape[0]

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        x_embedding = SequentialEmbeddingDiscrete(embedding)  # (n, 1, z_depth)
        h = x_embedding(input_x)  # (n, 1, z_depth)
        z = Reshape((self.z_depth,))(h)  # (n, z_depth)

        layer = SkipgramValidationFlatLayer(z_k=z_k,
                                            z_depth=self.z_depth,
                                            y_k=x_k,
                                            embeddings_regularizer=embeddings_regularizer)

        nll = layer([z, input_y])  # (n, z_depth)
        loss = Lambda(lambda _x: T.sum(_x, axis=1, keepdims=True),
                      output_shape=lambda _x: (_x[0], 1))(nll)

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

    def on_epoch_end(self, output_path, epoch):
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
