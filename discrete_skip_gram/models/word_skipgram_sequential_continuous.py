from keras.layers import Input, Embedding, Dense, Lambda, Activation
from theano import tensor as T
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from ..layers.utils import drop_dim_2, softmax_nd_layer
from ..layers.time_distributed_dense import TimeDistributedDense
from ..layers.encoder_lstm_continuous import EncoderLSTMContinuous
from keras import backend as K


class WordSkipgramSequentialContinuous(object):
    def __init__(self, dataset, hidden_dim, z_depth, z_k, schedule, lr=1e-4):
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.z_depth = z_depth
        self.z_k = z_k
        assert (len(schedule.shape) == 1)
        assert (schedule.shape[0] == z_depth)

        k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        # encoder
        embedding = Embedding(k, hidden_dim)
        embedded_x = drop_dim_2()(embedding(input_x))
        encoder = EncoderLSTMContinuous(z_depth, z_k, hidden_dim)
        z = encoder(embedded_x)  # n, z_depth, z_k

        # decoder
        lstm = LSTM(hidden_dim, return_sequences=True)
        h1 = TimeDistributedDense(hidden_dim)
        d = TimeDistributedDense(k)
        act = Activation('tanh')
        sm = softmax_nd_layer()
        p = sm(d(act(h1(lstm(z)))))

        # loss calculation
        weights = K.variable(schedule, name="schedule", dtype='float32')
        nll = Lambda(lambda (_p, _w): T.sum(-_w.dimshuffle(('x', 0, 'x')) * T.log(_p), axis=1),
                     output_shape=lambda (_p, _w): (_p[0], _p[2]))([p, weights])
        loss = Lambda(lambda (_nll, _y): T.reshape(_nll[T.arange(_nll.shape[0]), T.flatten(_y)], (-1, 1)),
                      output_shape=lambda (_nll, _y): (_nll[0], 1))([nll, input_y])

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[loss])
        self.model.compile(opt, loss_f)

    def train(self, window, batch_size, epochs, steps_per_epoch, **kwargs):
        gen = self.dataset.bigram_generator(n=batch_size, window=window)
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)
