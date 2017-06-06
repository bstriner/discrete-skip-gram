"""
Each element of sequence is an embedding layer
"""
import csv
import os

import numpy as np
from theano import tensor as T

from discrete_skip_gram.layers.utils import leaky_relu
from keras.layers import Input, Embedding, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from ..layers.uniform_smoothing import UniformSmoothing
from ..layers.unrolled.bias_layer import BiasLayer
from ..layers.utils import softmax_nd_layer
from ..skipgram_models.skipgram_model import SkipgramModel
from ..layers.one_bit_layer import OneBitLayer


class OneBitDiscreteModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 window,
                 z_k,
                 lr=1e-4,
                 inner_activation=leaky_relu,
                 kernel_regularizer=None,
                 embeddings_regularizer=None,
                 hidden_layers=2,
                 ):
        self.dataset = dataset
        self.window = window
        self.hidden_layers = hidden_layers
        self.z_k = z_k
        self.inner_activation = inner_activation
        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        # posterior p(y|x)
        # p(z|x)
        x_embedding = Embedding(input_dim=self.dataset.k,
                                output_dim=z_k,
                                embeddings_regularizer=embeddings_regularizer)
        h = x_embedding(input_x)
        h = Reshape((z_k,))(h)
        h = softmax_nd_layer()(h)
        p_z_given_x = UniformSmoothing()(h)  # n, z_k

        # NLL
        layer = OneBitLayer(y_k=x_k, z_k=z_k)
        prior_nll, posterior_nll = layer([p_z_given_x, input_y])

        # argmax
        z_sampled = Lambda(lambda _z: T.argmax(_z, axis=1, keepdims=True),
                           output_shape=lambda _z: (_z[0], 1))(p_z_given_x)
        z_embedding = Embedding(input_dim=z_k, output_dim=x_k)
        h = z_embedding(z_sampled)
        h = Reshape((x_k,))(h)
        h = softmax_nd_layer()(h)
        p_discrete = UniformSmoothing()(h)
        val_nll = Lambda(lambda (_p, _y): T.reshape(-T.log(_p[T.arange(_p.shape[0]), T.flatten(_y)]), (-1, 1)),
                         output_shape=lambda (_p, _y): (_p[0], 1))([p_discrete, input_y])

        # combine losses
        total_loss = Lambda(lambda (_a, _b, _c): _a + _b + _c,
                            output_shape=lambda (_a, _b, _c): _a)([posterior_nll, prior_nll, val_nll])

        #        self.loss_weight = K.variable(np.float32(loss_weight), dtype='float32', name='loss_weight')

        def avg_prior_nll(ytrue, ypred):
            return T.mean(prior_nll, axis=None)

        def avg_posterior_nll(ytrue, ypred):
            return T.mean(posterior_nll, axis=None)

        def avg_val_nll(ytrue, ypred):
            return T.mean(val_nll, axis=None)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[total_loss])
        self.model.compile(opt, loss_f, metrics=[avg_prior_nll, avg_posterior_nll, avg_val_nll])
        self.weights = self.model.weights + opt.weights

        # Encoder model
        self.model_encode = Model([input_x], z_sampled)

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
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
