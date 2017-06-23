"""
Each element of sequence is an embedding layer
"""

from theano import tensor as T

from discrete_skip_gram.layers.utils import leaky_relu
from keras.layers import Input, Embedding, Lambda, Reshape
from keras.models import Model
from .skipgram_model import SkipgramModel
from ..layers.skipgram_loookahead_flat_layer import SkipgramLookaheadFlatLayer
from ..layers.skipgram_loookahead_layer import SkipgramLookaheadLayer
from ..layers.skipgram_loookahead_partial_layer import SkipgramLookaheadPartialLayer
from ..layers.uniform_smoothing import UniformSmoothing
from ..layers.utils import nll_metrics
from ..layers.utils import softmax_nd_layer


class SkipgramDiscreteLookaheadModel(SkipgramModel):
    def __init__(self,
                 dataset,
                 units,
                 embedding_units,
                 window,
                 z_depth,
                 lookahead_depth,
                 z_k,
                 schedule,
                 opt,
                 mode=0,
                 inner_activation=leaky_relu,
                 kernel_regularizer=None,
                 embeddings_regularizer=None,
                 floating='float64',
                 hidden_layers=2,
                 layernorm=False,
                 batchnorm=True
                 ):
        assert z_depth >= lookahead_depth
        self.dataset = dataset
        self.floating = floating
        self.units = units
        self.embedding_units = embedding_units
        self.window = window
        self.hidden_layers = hidden_layers
        self.z_depth = z_depth
        self.lookahead_depth = lookahead_depth
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
                                embeddings_regularizer=embeddings_regularizer,
                                dtype=self.floating)
        h = x_embedding(input_x)
        h = Reshape((z_depth, z_k))(h)
        h = softmax_nd_layer()(h)
        h = UniformSmoothing()(h)
        p_z_given_x = h  # (n, z_depth, z_k)

        # skipgram
        if mode == 0:
            print "Sampled Lookahead ({}/{})".format(lookahead_depth, z_depth)
            skipgram = SkipgramLookaheadPartialLayer(y_k=x_k,
                                                     z_k=z_k,
                                                     z_depth=z_depth,
                                                     lookahead_depth=lookahead_depth,
                                                     units=units,
                                                     hidden_layers=hidden_layers,
                                                     embedding_units=embedding_units,
                                                     kernel_regularizer=kernel_regularizer,
                                                     inner_activation=inner_activation,
                                                     layernorm=layernorm,
                                                     batchnorm=batchnorm)
        elif mode == 1:
            print "Full Lookahead ({}/{})".format(lookahead_depth, z_depth)
            skipgram = SkipgramLookaheadLayer(y_k=x_k,
                                              z_k=z_k,
                                              z_depth=z_depth,
                                              units=units,
                                              hidden_layers=hidden_layers,
                                              embedding_units=embedding_units,
                                              lookahead_depth=lookahead_depth,
                                              kernel_regularizer=kernel_regularizer,
                                              inner_activation=inner_activation,
                                              layernorm=layernorm,
                                              batchnorm=batchnorm)
        elif mode == 2:
            print "Flat lookahead"
            skipgram = SkipgramLookaheadFlatLayer(y_k=x_k,
                                                  z_k=z_k,
                                                  z_depth=z_depth,
                                                  units=units,
                                                  floating=self.floating,
                                                  hidden_layers=hidden_layers,
                                                  embedding_units=embedding_units,
                                                  embeddings_regularizer=embeddings_regularizer,
                                                  lookahead_depth=lookahead_depth,
                                                  kernel_regularizer=kernel_regularizer,
                                                  inner_activation=inner_activation,
                                                  layernorm=layernorm,
                                                  batchnorm=batchnorm)
        else:
            raise ValueError("Unknown mode")

        losses = skipgram([p_z_given_x, input_y])
        schedule_var = T.constant(schedule, dtype=self.floating)
        weighted_losses = Lambda(lambda _x: _x * schedule_var,
                                 output_shape=lambda _x: _x)(losses)
        loss = Lambda(lambda _x: T.sum(_x, axis=1, keepdims=True),
                      output_shape=lambda _x: (_x[0], 1))(weighted_losses)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        avg_nll = T.mean(losses, axis=0)
        metrics = nll_metrics(avg_nll, z_depth)

        self.model = Model(inputs=[input_x, input_y], outputs=[loss])
        self.model.compile(opt, loss_f, metrics=metrics)
        self.model._make_train_function()
        self.weights = self.model.weights + opt.weights

        # Encoder model
        z_sampled = Lambda(lambda _x: T.argmax(_x, axis=-1),
                           output_shape=lambda _x: (_x[0], _x[1]))(p_z_given_x)
        self.model_encode = Model([input_x], z_sampled)

    def on_epoch_end(self, output_path, epoch):
        self.write_encodings("{}/encodings-{:08d}".format(output_path, epoch))
        self.save("{}/model-{:08d}.h5".format(output_path, epoch))
