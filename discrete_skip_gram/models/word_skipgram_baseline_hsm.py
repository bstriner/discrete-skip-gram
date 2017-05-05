import csv
import os

import numpy as np
from keras.layers import Input, Embedding
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .sg_model import SGModel
from ..layers.sequential_embedding_discrete import SequentialEmbeddingDiscrete
from ..layers.skipgram_hsm_layer import SkipgramHSMLayer, SkipgramHSMPolicyLayer
from ..layers.utils import drop_dim_2


class WordSkipgramBaselineHSM(SGModel):
    def __init__(self, dataset, units, window,
                 hsm,
                 embedding_units,
                 kernel_regularizer=None,
                 lr=1e-4):
        self.dataset = dataset
        self.units = units
        self.embedding_units=embedding_units
        self.window = window
        self.y_depth = window * 2
        self.hsm = hsm

        k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((self.y_depth,), dtype='int32', name='input_y')

        x_embedding = Embedding(k, embedding_units, embeddings_regularizer=kernel_regularizer)
        z = drop_dim_2()(x_embedding(input_x))
        skipgram = SkipgramHSMLayer(units=units,
                                    kernel_regularizer=kernel_regularizer,
                                    embeddings_regularizer=kernel_regularizer)

        y_embedding = SequentialEmbeddingDiscrete(self.hsm.codes)
        y_embedded = y_embedding(input_y)

        nll = skipgram([z, y_embedded])

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def avg_nll(ytrue, ypred):
            return T.mean(nll, axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[nll])
        self.model.compile(opt, loss_f, metrics=[avg_nll])
        self.weights = self.model.weights + opt.weights

        self.model_encode = Model(inputs=[input_x], outputs=[z])

        srng = RandomStreams(123)
        policy = SkipgramHSMPolicyLayer(skipgram, srng=srng, y_depth=self.y_depth, code_depth=self.hsm.codes.shape[1])
        ypred = policy(z)
        self.model_predict = Model(inputs=[input_x], outputs=[ypred])

    def summary(self):
        print "Skipgram Model"
        self.model.summary()
        print "Skipgram Policy Model"
        self.model_predict.summary()

    def write_encodings(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        x = np.arange(self.dataset.k).reshape((-1, 1))
        z = self.model_encode.predict(x, verbose=0)
        np.save(output_path + ".npy", z)

    def on_epoch_end(self, output_path, epoch):
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.write_predictions("{}/predictions-{:08d}.csv".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
