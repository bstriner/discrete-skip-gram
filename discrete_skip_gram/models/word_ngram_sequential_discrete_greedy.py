from keras.layers import Input, Embedding, Dense, Lambda, Activation
from theano import tensor as T
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
from ..layers.ngram_layer import NgramLayer, NgramLayerGenerator
from ..layers.utils import drop_dim_2
from theano.tensor.shared_randomstreams import RandomStreams
import keras.backend as K
from discrete_skip_gram.layers.encoder_lstm_deterministic import EncoderLSTMDeterministic
from discrete_skip_gram.layers.ngram_layer_distributed import NgramLayerDistributed
from discrete_skip_gram.layers.decoder_lstm_skipgram import DecoderLSTMSkipgram


class WordNgramSequentialDiscreteGreedy(object):
    def __init__(self, dataset,
                 hidden_dim=256, window=3, lr=1e-3, z_depth=6, z_k=4,
                 reg=None):
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.window = window
        self.z_depth = z_depth
        self.z_k = z_k
        k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((window * 2,), dtype='int32', name='input_y')

        embedding = Embedding(k, hidden_dim, embeddings_regularizer=reg)
        embedded = drop_dim_2()(embedding(input_x))
        encoder = EncoderLSTMDeterministic(k=self.z_k, depth=self.z_depth, units=self.hidden_dim,
                                           kernel_regularizer=reg)
        pz, z = encoder(embedded)  # n, z_depth, z_k
        #pz: n, z_depth, z_k (float32)
        #z: n, z_depth (int)
        hlstm = DecoderLSTMSkipgram(z_k=z_k, y_k=k, units=self.hidden_dim)
        nll = hlstm([z, input_y]) # n, z_depth, z_k

        weighted_loss = Lambda(lambda (_nll, _pz): T.sum(T.sum(_nll*_pz,axis=1), axis=1, keepdims=True),
                               output_shape=lambda (_nll, _pz): (_nll[0],1))([nll, pz])

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[weighted_loss])
        self.model.compile(opt, loss_f)

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
