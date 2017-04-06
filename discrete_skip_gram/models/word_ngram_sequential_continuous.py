from keras.layers import Input, Embedding, Dense, Lambda, Activation
from theano import tensor as T
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
from ..layers.ngram_layer import NgramLayer, NgramLayerGenerator
from ..layers.utils import drop_dim_2
from theano.tensor.shared_randomstreams import RandomStreams
import keras.backend as K
from discrete_skip_gram.layers.encoder_lstm_continuous import EncoderLSTMContinuous
from discrete_skip_gram.layers.ngram_layer_distributed import NgramLayerDistributed


class WordNgramSequentialContinuous(object):
    def __init__(self, dataset, schedule, hidden_dim=256, window=3, lr=1e-3, z_depth=8, z_k=2):
        self.dataset = dataset
        self.schedule = schedule
        self.hidden_dim = hidden_dim
        self.window = window
        self.z_depth = z_depth
        self.z_k = z_k
        k = self.dataset.k
        assert (len(schedule.shape) == 1)
        assert (schedule.shape[0] == z_depth)

        sched = K.variable(schedule, dtype='float32', name='schedule')

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((window * 2,), dtype='int32', name='input_y')

        embedding = Embedding(k, hidden_dim)
        embedded = drop_dim_2()(embedding(input_x))
        encoder = EncoderLSTMContinuous(z_depth=z_depth, z_k=z_k, units=self.hidden_dim)
        z = encoder(embedded)  # n, z_depth, z_k
        hlstm = LSTM(self.hidden_dim, return_sequences=True)
        h = hlstm(z)
        ngram_layer = NgramLayerDistributed(k=k, units=self.hidden_dim)
        nll_partial = ngram_layer([h, input_y])  # n, z_depth, window*2
        nll = Lambda(lambda _x: T.mean(_x, axis=2), output_shape=lambda _x: (_x[0], _x[1]))(nll_partial)  # n, z_depth

        weighted_loss = Lambda(lambda _nll: T.sum(_nll * (sched.dimshuffle(('x', 0))), axis=1, keepdims=True),
                               output_shape=lambda _nll: (_nll[0], 1))(nll)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def nll_initial(ytrue, ypred):
            return T.mean(nll[:, 0], axis=0)

        def nll_final(ytrue, ypred):
            return T.mean(nll[:, -1], axis=0)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[weighted_loss])
        self.model.compile(opt, loss_f, metrics=[nll_initial, nll_final])

        srng = RandomStreams(123)
        rng = Lambda(lambda _x: srng.uniform(low=0, high=1, size=(_x.shape[0], window * 2), dtype='float32'),
                     output_shape=lambda _x: (_x[0], window * 2))(input_x)
        gen = NgramLayerGenerator(ngram_layer)
        zpart = Lambda(lambda _x: _x[:, -1, :], output_shape=lambda _x: (_x[0], _x[2]))(h)
        y_gen = gen([zpart, rng])
        self.model_predict = Model(inputs=[input_x], outputs=[y_gen])

        self.model_encode = Model(inputs=[input_x], outputs=[z])

    def train(self, batch_size, epochs, steps_per_epoch, **kwargs):
        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)
