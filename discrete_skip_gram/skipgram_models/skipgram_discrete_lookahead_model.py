"""
Each element of sequence is an embedding layer
"""

from theano import tensor as T

from discrete_skip_gram.layers.utils import leaky_relu
from keras.layers import Input, Embedding, Lambda, Reshape
from keras.models import Model
from .skipgram_model import SkipgramModel
from ..layers.skipgram_loookahead_layer import SkipgramLookaheadLayer
from ..layers.utils import nll_metrics
from ..layers.utils import softmax_nd_layer


class SkipgramDiscreteLookaheadModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 units,
                 embedding_units,
                 window,
                 z_depth,
                 z_k,
                 schedule,
                 opt,
                 inner_activation=leaky_relu,
                 kernel_regularizer=None,
                 embeddings_regularizer=None,
                 hidden_layers=2,
                 layernorm=False,
                 batchnorm=True,
                 ):
        self.dataset = dataset
        self.units = units
        self.embedding_units = embedding_units
        self.window = window
        self.hidden_layers = hidden_layers
        self.z_depth = z_depth
        self.z_k = z_k
        self.inner_activation = inner_activation
        self.schedule = schedule
        assert z_depth > 0
        assert schedule.ndim == 1
        assert schedule.shape[0] == z_depth
        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        # p(z|x)
        x_embedding = Embedding(input_dim=self.dataset.k,
                                output_dim=z_depth * z_k,
                                embeddings_regularizer=embeddings_regularizer)
        h = x_embedding(input_x)
        h = Reshape((z_depth, z_k))(h)
        h = softmax_nd_layer()(h)
        # h = UniformSmoothing()(h)
        p_z_given_x = h  # (n, z_depth, z_k)

        # skipgram
        skipgram = SkipgramLookaheadLayer(y_k=x_k,
                                          z_k=z_k,
                                          z_depth=z_depth,
                                          units=units,
                                          hidden_layers=hidden_layers,
                                          embedding_units=embedding_units,
                                          kernel_regularizer=kernel_regularizer,
                                          inner_activation=inner_activation,
                                          layernorm=layernorm,
                                          batchnorm=batchnorm)
        losses = skipgram([p_z_given_x, input_y])

        weighted_losses = Lambda(lambda _x: _x * schedule,
                                 output_shape=lambda _x: _x)(losses)
        loss = Lambda(lambda _x: T.sum(_x, axis=1, keepdims=True),
                      output_shape=lambda _x: (_x[0], 1))(weighted_losses)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        avg_nll = T.mean(losses, axis=0)
        metrics = nll_metrics(avg_nll, z_depth)

        self.model = Model(inputs=[input_x, input_y], outputs=[loss])
        self.model.compile(opt, loss_f, metrics=metrics)
        self.weights = self.model.weights + opt.weights

        # Encoder model
        z_sampled = Lambda(lambda _x: T.argmax(_x, axis=-1),
                           output_shape=lambda _x: (_x[0], _x[1]))(p_z_given_x)
        self.model_encode = Model([input_x], z_sampled)

    def on_epoch_end(self, output_path, epoch):
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
