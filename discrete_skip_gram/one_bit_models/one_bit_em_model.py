"""
Each element of sequence is an embedding layer
"""
import csv
import os

import numpy as np
from theano import tensor as T
from theano.gradient import zero_grad

from keras import backend as K
from keras.layers import Input, Lambda
from keras.layers import Reshape, Embedding
from keras.models import Model
from keras.optimizers import Adam
from ..layers.nll_layer import NLL
from ..layers.sequential_embedding_discrete import SequentialEmbeddingDiscrete
from ..layers.uniform_smoothing import UniformSmoothing
from ..layers.unrolled.bias_layer import BiasLayer
from ..layers.utils import softmax_nd_layer
from ..skipgram_models.skipgram_model import SkipgramModel


class OneBitEMModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 window,
                 lr=1e-4):
        self.dataset = dataset
        self.window = window
        x_k = self.dataset.k
        self.x_k = x_k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        # x embedding into z
        order = np.arange(0, x_k)
        np.random.shuffle(order)
        mid = int(x_k / 2)
        embedding = np.zeros((x_k, 1))
        embedding[order[mid:], :] = 1
        embedding_layer = SequentialEmbeddingDiscrete(embedding=embedding)
        z = Reshape((1,))(embedding_layer(input_x))
        self.embedding_var = embedding_layer.embedding

        # p(y|z)
        y_layer = BiasLayer(x_k * 2)
        h = y_layer(input_x)
        h = Reshape((2, x_k))(h)
        p_y_given_z = UniformSmoothing()(softmax_nd_layer()(h))  # (n, 2, x_k)

        p_y_given_z_at_y = Lambda(lambda (_p, _y): _p[T.arange(_p.shape[0]), :, T.flatten(_y)],
                                  output_shape=lambda (_p, _y): (_p[0], _p[1]))([p_y_given_z, input_y])  # (n, 2)
        nllpart = NLL()(p_y_given_z_at_y)
        p_y_given_z_at_yx = Lambda(lambda (_p, _z): T.reshape(_p[T.arange(_p.shape[0]), T.flatten(_z)], (-1, 1)),
                                   output_shape=lambda (_p, _z): (_p[0], 1))([p_y_given_z_at_y, z])
        nll = NLL()(p_y_given_z_at_yx)

        # loss per word
        loss_embedding_layer = Embedding(input_dim=self.x_k, output_dim=2)
        loss_per_word = Reshape((2,))(loss_embedding_layer(input_x))
        loss_pred = Lambda(lambda (_a, _b): T.sum(T.square(_a - zero_grad(_b)), axis=1, keepdims=True),
                           output_shape=lambda (_a, _b): (_a[0], 1))([loss_per_word, nllpart])

        tot_loss = Lambda(lambda (_a, _b): _a + _b,
                          output_shape=lambda (_a, _b): _a)(
            [nll, loss_pred])

        def avg_posterior_nll(ytrue, ypred):
            return T.mean(nll, axis=None)

        def avg_loss_pred(ytrue, ypred):
            return T.mean(loss_pred, axis=None)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[tot_loss])
        self.model.compile(opt, loss_f, metrics=[avg_loss_pred, avg_posterior_nll])
        self.weights = self.model.weights + opt.weights

        # Encoder model
        self.model_encode = Model([input_x], z)
        loss_diff = Lambda(lambda _a: T.reshape((_a[:, 0]) - (_a[:, 1]), (-1, 1)),
                           output_shape=lambda _a: (_a[0], 1))(loss_per_word)
        self.model_loss = Model([input_x], loss_diff)

    def write_encodings(self, output_path):
        x = np.arange(self.dataset.k).reshape((-1, 1))
        z = self.model_encode.predict(x, verbose=0)

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path + ".csv", 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word", "Encoding"])
            for i in range(self.dataset.k):
                enc = z[i, 0]
                word = self.dataset.get_word(i)
                w.writerow([i, word, enc])
        np.save(output_path + ".npy", z)

    def on_epoch_end(self, output_path, epoch):
        # reassign clusters
        x = np.reshape(np.arange(self.x_k), (-1, 1))
        loss_diff = self.model_loss.predict(x, verbose=0)  # (n,1)
        order = np.argsort(loss_diff[:, 0])
        embedding = np.zeros((self.x_k, 1))
        mid = int(self.x_k / 2)
        idx = order[mid:]
        embedding[idx, :] = 1
        K.set_value(self.embedding_var, embedding)
        # write results
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
