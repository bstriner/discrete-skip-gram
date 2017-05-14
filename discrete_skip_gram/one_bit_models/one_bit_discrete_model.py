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
from ..skipgram_models.skipgram_model import SkipgramModel
from ..layers.dense_batch import DenseBatch
from ..layers.highway_layer import HighwayLayer
from ..layers.shift_padding_layer import ShiftPaddingLayer
from ..layers.unrolled.bias_layer import BiasLayer


class OneBitDiscreteModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 units,
                 embedding_units,
                 window,
                 z_k,
                 lr=1e-4,
                 lr_a=1e-3,
                 inner_activation=leaky_relu,
                 kernel_regularizer=None,
                 embeddings_regularizer=None,
                 hidden_layers=2,
                 layernorm=False,
                 loss_weight=1e-2,
                 adversary_weight=1.0
                 ):
        self.dataset = dataset
        self.window = window
        self.hidden_layers = hidden_layers
        self.z_k = z_k
        self.inner_activation = inner_activation
        srng = RandomStreams(123)
        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        x_embedding = Embedding(input_dim=self.dataset.k,
                                output_dim=z_k,
                                embeddings_regularizer=embeddings_regularizer)
        rs = Reshape((z_k,))
        sm = softmax_nd_layer()
        z = sm(rs(x_embedding(input_x)))  # n, z_k

        h = BiasLayer(units=z_k * x_k, bias_regularizer=kernel_regularizer)(input_x)  # n, z_k*x_k
        h = Reshape((z_k, x_k))(h)
        p = softmax_nd_layer()(h)  # n, z_k, x_k

        eps = 1e-7
        scale = 1.0 - (eps * x_k)
        nll = Lambda(lambda (_p, _y): -T.log(eps + (scale*_p[T.arange(_p.shape[0]), :, T.flatten(_y)])),
                     output_shape=lambda (_p, _y): (_p[0], _p[1]))([p, input_y])  # n, z_k

        loss = Lambda(lambda (_nll, _z): T.sum(_nll * _z, axis=1, keepdims=True),
                      output_shape=lambda (_nll, _z): (_nll[0], 1),
                      name="loss_layer")([nll, z])

        #        self.loss_weight = K.variable(np.float32(loss_weight), dtype='float32', name='loss_weight')

        def avg_nll(ytrue, ypred):
            return T.mean(loss, axis=None)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def discrete_nll(ytrue, ypred):
            zd = T.argmax(z, axis=1)  # (n,)
            nlld = nll[T.arange(nll.shape[0]), zd]
            return T.mean(nlld, axis=None)

        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[loss])
        self.model.compile(opt, loss_f, metrics=[avg_nll, discrete_nll])
        self.weights = self.model.weights + opt.weights

        # Encoder model
        z_sampled = Lambda(lambda _z: T.argmax(_z, axis=1, keepdims=True),
                           output_shape=lambda _z: (_z[0], 1))(z)
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
