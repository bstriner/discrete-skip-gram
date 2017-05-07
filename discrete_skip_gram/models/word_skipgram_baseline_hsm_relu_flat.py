import csv
import os
import numpy as np
from keras.layers import Input, Embedding
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .sg_model import SGModel
from ..layers.sequential_embedding_discrete import SequentialEmbeddingDiscrete
from ..layers.skipgram_hsm_layer_relu_flat import SkipgramHSMLayerReluFlat, SkipgramHSMPolicyLayerReluFlat
from ..layers.utils import drop_dim_2, shift_tensor_layer


class WordSkipgramBaselineHSMReluFlat(SGModel):
    def __init__(self,
                 dataset,
                 units,
                 window,
                 hsm,
                 embedding_units,
                 layernorm=True,
                 kernel_regularizer=None,
                 inner_activation=T.nnet.relu,
                 embeddings_regularizer=None,
                 lr=1e-4):
        self.dataset = dataset
        self.units = units
        self.embedding_units = embedding_units
        self.window = window
        self.y_depth = window * 2
        self.hsm = hsm
        self.layernorm=layernorm
        k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((self.y_depth,), dtype='int32', name='input_y')

        x_embedding = Embedding(k, embedding_units,
                                embeddings_regularizer=embeddings_regularizer)
        z = drop_dim_2()(x_embedding(input_x))
        skipgram = SkipgramHSMLayerReluFlat(units=units,
                                            k=k,
                                            layernorm=layernorm,
                                            embedding_units=embedding_units,
                                            inner_activation=inner_activation,
                                            kernel_regularizer=kernel_regularizer,
                                            embeddings_regularizer=embeddings_regularizer)

        y0 = shift_tensor_layer()(input_y)
        y_embedding = SequentialEmbeddingDiscrete(self.hsm.codes)
        y1 = y_embedding(input_y)
        nll = skipgram([z, y0, y1])

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
        policy = SkipgramHSMPolicyLayerReluFlat(skipgram, srng=srng,
                                                y_depth=self.y_depth,
                                                wordcodes=hsm.words,
                                                code_depth=self.hsm.codes.shape[1])
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
        self.write_predictions_flat("{}/predictions-{:08d}.csv".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
