from keras.layers import Input, Embedding, Dense, Lambda, Activation
from theano import tensor as T
from keras.models import Model
from keras.optimizers import Adam
from ..layers.ngram_layer import NgramLayer
from ..layers.utils import drop_dim_2


class WordNgramWithContext(object):
    def __init__(self, dataset, hidden_dim, window, lr=1e-4):
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.window = window
        k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((window * 2,), dtype='int32', name='input_x')


        embedding = Embedding(k, hidden_dim)
        z = drop_dim_2()(embedding(input_x))
        ngram_layer = NgramLayer(k, self.hidden_dim)
        nll_partial = ngram_layer([z, input_y])
        nll = Lambda(lambda _x: T.mean(_x, axis=1), output_shape=lambda _x: (_x[0], _x[2]))(nll_partial)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[nll])
        self.model.compile(opt, loss_f)

    def train(self, batch_size, epochs, steps_per_epoch, **kwargs):
        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)
