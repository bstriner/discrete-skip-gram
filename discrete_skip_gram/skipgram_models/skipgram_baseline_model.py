"""
Each element of sequence is an embedding layer
"""
import os

import numpy as np
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_skip_gram.layers.utils import leaky_relu
from keras.layers import Input, Embedding, Lambda, Dense
from keras.models import Model
from keras.optimizers import Adam
from .skipgram_model import SkipgramModel
from ..layers.nll_layer import NLL
from ..layers.uniform_smoothing import UniformSmoothing
from ..layers.unrolled.sampler_layer import SamplerLayer
from ..layers.utils import drop_dim_2


class SkipgramBaselineModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 embedding_units,
                 window,
                 lr=1e-4,
                 inner_activation=leaky_relu,
                 kernel_regularizer=None,
                 embeddings_regularizer=None,
                 hidden_layers=2
                 ):
        self.dataset = dataset
        self.window = window
        self.hidden_layers = hidden_layers
        self.inner_activation = inner_activation

        srng = RandomStreams(123)
        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        x_embedding = Embedding(input_dim=self.dataset.k,
                                output_dim=embedding_units,
                                embeddings_regularizer=embeddings_regularizer)
        z = drop_dim_2()(x_embedding(input_x))  # (n, embedding_units)

        p = Dense(x_k,
                  kernel_regularizer=kernel_regularizer,
                  activation='softmax')(z)  # (n, y_k)
        p = UniformSmoothing()(p)
        p_t = Lambda(lambda (_p, _y): T.reshape(_p[T.arange(_p.shape[0]), T.flatten(_y)], (-1, 1)),
                     output_shape=lambda (_p, _y): (_p[0], 1))([p, input_y])
        nll = NLL()(p_t)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def avg_nll(ytrue, ypred):
            return T.mean(nll, axis=None)

        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[nll])
        self.model.compile(opt, loss_f, metrics=[avg_nll])
        self.model._make_train_function()
        self.weights = self.model.weights + opt.weights
        policy_sampler = SamplerLayer(srng=srng)
        ygen = policy_sampler(p)
        self.model_predict = Model([input_x], ygen)

        # Encoder model
        self.model_encode = Model([input_x], z)

        # prob: (n, z_depth, x_k)
        self.model_probability = Model([input_x], p)

    def write_encodings(self, output_path):
        x = np.arange(self.dataset.k).reshape((-1, 1))
        z = self.model_encode.predict(x, verbose=0)

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        np.save(output_path + ".npy", z)

    def on_epoch_end(self, output_path, epoch):
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
