from keras.layers import Input, Embedding, Dense, Lambda, Activation
from theano import tensor as T
from keras.models import Model
from keras.optimizers import Adam


class WordUnigram(object):
    def __init__(self, dataset, lr=1e-4):
        self.dataset = dataset

        k = self.dataset.k

        y = Input((1,), dtype='int32')
        ones = Lambda(lambda _y: T.ones((_y.shape[0], 1), dtype='float32'),
                      output_shape=lambda _y: (_y[0], 1))
        d = Dense(k)
        sm = Activation('softmax')
        p = sm(d(ones(y)))
        loss = Lambda(lambda (_p, _y): T.reshape(-T.log(_p[T.arange(_p.shape[0]), T.flatten(_y)]), (-1, 1)),
                      output_shape=lambda (_p, _y): (_p[0], 1))([p, y])

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[y], outputs=[loss])
        self.model.compile(opt, loss_f)

    def train(self, batch_size, epochs, steps_per_epoch, **kwargs):
        gen = self.dataset.unigram_generator(batch_size)
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)
