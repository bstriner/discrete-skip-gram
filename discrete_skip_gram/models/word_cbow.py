from keras.layers import Input, Embedding, Dense, Lambda, Activation
from theano import tensor as T
from keras.models import Model
from keras.optimizers import Adam
from ..layers.utils import drop_dim_2
from keras.layers import Flatten
class WordCbow(object):
    def __init__(self, dataset, hidden_dim, window, lr=1e-4):
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.window = window

        k = self.dataset.k

        input_x = Input((window*2,), dtype='int32', name='input_x')
        input_y = Input((1,), dtype='int32', name='input_y')

        embedding = Embedding(k, hidden_dim)
        flatten = Flatten()
        d1 = Dense(hidden_dim)
        act = Activation('tanh')
        z = act(d1(flatten(embedding(input_x))))


        d2 = Dense(hidden_dim)
        act2 = Activation('tanh')
        d3 = Dense(k)
        sm = Activation('softmax')
        p = sm(d3(act2(d2(z))))

        loss = Lambda(lambda (_p, _y): T.reshape(-T.log(_p[T.arange(_p.shape[0]), T.flatten(_y)]), (-1, 1)),
                      output_shape=lambda (_p, _y): (_p[0], 1))([p, input_y])

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        opt = Adam(lr)
        self.model = Model(inputs=[input_x, input_y], outputs=[loss])
        self.model.compile(opt, loss_f)

    def train(self, batch_size, epochs, steps_per_epoch, **kwargs):
        gen = self.dataset.cbow_generator(n=batch_size, window=self.window)
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)
