from keras.layers import Input, Embedding, Dense, Lambda, Activation
from theano import tensor as T
from keras.models import Model
from keras.optimizers import Adam
from ..layers.ngram_layer import NgramLayer, NgramLayerGenerator
from theano.tensor.shared_randomstreams import RandomStreams


class WordNgramNoContext(object):
    def __init__(self, dataset, hidden_dim, window, lr=1e-4):
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.window = window
        k = self.dataset.k

        input_x = Input((window * 2,), dtype='int32', name='input_x')
        z = Lambda(lambda _x: T.zeros((_x.shape[0], 1), dtype='float32'), output_shape=lambda _x: (_x[0], 1))(input_x)
        ngram_layer = NgramLayer(k, self.hidden_dim)
        nll_partial = ngram_layer([z, input_x])
        nll = Lambda(lambda _x: T.mean(_x, axis=1, keepdims=True), output_shape=lambda _x: (_x[0], 1))(nll_partial)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x], outputs=[nll])
        self.model.compile(opt, loss_f)

        srng = RandomStreams(123)
        z_test = Input((1,), dtype='float32')
        rng = Lambda(lambda _x: srng.uniform(low=0, high=1, size=(_x.shape[0], window * 2), dtype='float32'),
                     output_shape=lambda _x: (_x[0], window * 2))(z_test)
        gen = NgramLayerGenerator(ngram_layer)
        y_gen = gen([z_test, rng])
        self.model_predict = Model(inputs=[z_test], outputs=[y_gen])

    def train(self, batch_size, epochs, steps_per_epoch, **kwargs):
        gen = self.dataset.skipgram_generator_no_context(n=batch_size, window=self.window)
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)
