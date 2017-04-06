from keras.layers import Input, Embedding, Dense, Lambda, Activation
from theano import tensor as T
from keras.models import Model
from keras.optimizers import Adam
from ..layers.ngram_layer import NgramLayer, NgramLayerGenerator
from ..layers.utils import drop_dim_2
from theano.tensor.shared_randomstreams import RandomStreams


class WordNgramWithContext(object):
    def __init__(self, dataset, hidden_dim, window, lr=1e-4):
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.window = window
        k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((window * 2,), dtype='int32', name='input_y')

        embedding = Embedding(k, hidden_dim)
        z = drop_dim_2()(embedding(input_x))
        ngram_layer = NgramLayer(k, self.hidden_dim)
        nll_partial = ngram_layer([z, input_y])
        nll = Lambda(lambda _x: T.mean(_x, axis=1), output_shape=lambda _x: _x)(nll_partial)

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[nll])
        self.model.compile(opt, loss_f)

        srng = RandomStreams(123)
        rng = Lambda(lambda _x: srng.uniform(low=0, high=1, size=(_x.shape[0], window * 2), dtype='float32'),
                     output_shape=lambda _x: (_x[0], window * 2))(input_x)
        gen = NgramLayerGenerator(ngram_layer)
        y_gen = gen([z, rng])
        self.model_predict = Model(inputs=[input_x], outputs=[y_gen])

    def train(self, batch_size, epochs, steps_per_epoch, **kwargs):
        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)
