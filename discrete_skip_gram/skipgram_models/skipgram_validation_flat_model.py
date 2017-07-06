"""
Each element of sequence is an embedding layer
"""

import numpy as np
from theano import tensor as T

from keras.layers import Input, Lambda
from keras.layers import Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import L1L2
from .skipgram_model import SkipgramModel
from ..dataset_util import load_dataset
from ..layers.sequential_embedding_discrete import SequentialEmbeddingDiscrete
from ..layers.skipgram_validation_flat_layer import SkipgramValidationFlatLayer
from ..layers.utils import nll_metrics


class SkipgramValidationFlatModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 embedding,
                 window,
                 z_k,
                 lr=1e-4,
                 embeddings_regularizer=None,
                 ):
        self.dataset = dataset
        self.window = window
        self.z_depth = embedding.shape[1]
        self.z_k = z_k
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

        self.model = Model([input_x, input_y], loss)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        avg_nll = T.mean(nll, axis=0)
        metrics = nll_metrics(avg_nll, self.z_depth)
        opt = Adam(lr)

        self.model = Model(inputs=[input_x, input_y], outputs=[loss])
        self.model.compile(opt, loss_f, metrics=metrics)
        self.model._make_train_function()
        self.weights = self.model.weights + opt.weights

    def on_epoch_end(self, output_path, epoch):
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))


def validate_skipgram_flat(outputpath, embeddingpath):
    embedding = np.load(embeddingpath)
    ds = load_dataset()
    batch_size = 256
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 10
    z_k = 2
    lr = 3e-4
    embeddings_regularizer = L1L2(1e-9, 1e-9)
    model = SkipgramValidationFlatModel(dataset=ds,
                                        z_k=z_k,
                                        #embeddings_regularizer=embeddings_regularizer,
                                        embedding=embedding,
                                        window=window,
                                        lr=lr)
    model.summary()
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                output_path=outputpath,
                frequency=frequency)
