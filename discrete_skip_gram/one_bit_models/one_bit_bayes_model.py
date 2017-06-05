"""
Each element of sequence is an embedding layer
"""
import csv
import os

import numpy as np
from keras.layers import Input, Embedding, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T

from discrete_skip_gram.layers.utils import leaky_relu
from ..layers.one_bit_bayes_layer import OneBitBayesLayer
from ..layers.one_bit_validation_layer import OneBitValidationLayer
from ..layers.uniform_smoothing import UniformSmoothing
from ..layers.utils import softmax_nd_layer
from ..skipgram_models.skipgram_model import SkipgramModel


class OneBitBayesModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 window,
                 z_k,
                 lr=1e-4,
                 inner_activation=leaky_relu,
                 kernel_regularizer=None,
                 embeddings_regularizer=None,
                 hidden_layers=2):
        self.dataset = dataset
        self.window = window
        self.hidden_layers = hidden_layers
        self.z_k = z_k
        self.inner_activation = inner_activation
        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        # p(z|x)
        x_embedding = Embedding(input_dim=self.dataset.k,
                                output_dim=z_k,
                                embeddings_regularizer=embeddings_regularizer)
        h = x_embedding(input_x)
        h = Reshape((z_k,))(h)
        h = softmax_nd_layer()(h)
        p_z_given_x = UniformSmoothing()(h)  # (n, z_k)

        print "Tensors: {} and {}".format(p_z_given_x, input_y)

        # nll
        layer = OneBitBayesLayer(y_k=x_k, z_k=z_k)
        prior_nll, posterior_nll = layer([p_z_given_x, input_y])

        # validation nll
        val_layer = OneBitValidationLayer(y_k=x_k, z_k=z_k)
        val_nll = val_layer([p_z_given_x, input_y])  # (n,1)

        tot_loss = Lambda(lambda (_a, _b, _c): _a + _b+_c,
                          output_shape=lambda (_a, _b, _c): _a)(
            [val_nll, prior_nll, posterior_nll])

        def avg_prior_nll(ytrue, ypred):
            return T.mean(prior_nll, axis=None)

        def avg_posterior_nll(ytrue, ypred):
            return T.mean(posterior_nll, axis=None)

        def avg_val_nll(ytrue, ypred):
            return T.mean(val_nll, axis=None)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[tot_loss])
        self.model.compile(opt, loss_f, metrics=[avg_prior_nll, avg_posterior_nll, avg_val_nll])
        self.weights = self.model.weights + opt.weights

        # Encoder model
        z_sampled = Lambda(lambda _z: T.argmax(_z, axis=1, keepdims=True),
                           output_shape=lambda _z: (_z[0], 1))(p_z_given_x)
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
