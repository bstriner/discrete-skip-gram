from keras.layers import Input, Embedding, Dense, Activation, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.recurrent import LSTM
from ..layers.encoder_dqn import DQNEncoderValue, DQNEncoderPolicy
from ..layers.utils import softmax_nd_layer, custom_loss, rewards_to_values, drop_dim_2
import theano.tensor as T
import numpy as np
from tqdm import tqdm

from theano.tensor.shared_randomstreams import RandomStreams


class WordDQN(object):
    """Word-level sequential discrete encoder"""

    def __init__(self, z_depth, z_k, x_k, y_k, units, discount=0.8,
                 initial_exploration=0.1):
        self.z_depth = z_depth
        self.discount = discount

        input_x = Input((1,), dtype='int32', name="input_x")
        input_z = Input((z_depth,), dtype='int32', name="input_z")
        input_y = Input((1,), dtype='int32', name="input_y")
        opt_v = Adam(1e-3)
        opt_d = Adam(1e-3)

        # Value model
        embedding_x = Embedding(x_k, units)
        embedded_x = embedding_x(input_x)
        embedded_x = drop_dim_2()(embedded_x)

        value_layer = DQNEncoderValue(z_k, units)
        value = value_layer([embedded_x, input_z])
        self.model_value = Model(inputs=[input_x, input_z], outputs=[value])
        self.model_value.compile(opt_v, 'mean_absolute_error')

        # Stochastic policy
        policy_layer = DQNEncoderPolicy(value_layer, z_depth, initial_exploration)
        encoded_z = policy_layer(embedded_x)
        self.model_encoder = Model(inputs=[input_x], outputs=[encoded_z])

        # Deterministic policy
        encoded_z_deterministic = policy_layer(embedded_x, stochastic=False)
        self.model_encoder_deterministic = Model(inputs=[input_x], outputs=[encoded_z_deterministic])

        # Decoder model
        embedding_z = Embedding(z_k, units)
        lstm_z = LSTM(units, return_sequences=True)
        embedded_z = embedding_z(input_z)

        h = lstm_z(embedded_z)
        h = TimeDistributed(Dense(units))(h)
        h = Activation('tanh')(h)
        h = TimeDistributed(Dense(y_k))(h)
        y = softmax_nd_layer()(h)

        def decoder_loss((_y, _input_y)):
            tmp = T.log(_y)
            return tmp[T.arange(tmp.shape[0]), :, T.flatten(_input_y)]

        dloss = Lambda(decoder_loss, output_shape=lambda (_y, _input_y): (_y[0], _y[1]))([y, input_y])
        self.model_decoder = Model(inputs=[input_z, input_y], outputs=[dloss])
        self.model_decoder.compile(opt_d, custom_loss)

        """
        m = Model([input_z], [y])
        print "Test Y"
        _z = np.random.randint(0, z_k, (32, z_depth))
        _y = np.random.randint(0, y_k, (32,1))
        _ty = m.predict_on_batch([_z])
        print "Ty: {}".format(_ty.shape)
        m.summary()

        _loss = self.model_decoder.predict_on_batch([_z,_y])
        print "Test decoder"
        print "Loss: {}".format(_loss.shape)
        self.model_decoder.summary()
        # print "Y: {}".format(_y.shape)
        """

    def summary(self):
        print "Value"
        self.model_value.summary()
        print "Encoder"
        self.model_encoder.summary()
        print "Decoder"
        self.model_decoder.summary()

    def fit_generator(self, gen, epochs=1000, batches=256, samples=4, callback=None):
        for epoch in tqdm(range(epochs), desc="Training"):
            for _ in tqdm(range(batches), desc="Epoch {}".format(epoch)):
                # get samples
                samps = [next(gen) for _ in range(samples)]
                # encode samples
                encoded = [self.model_encoder.predict_on_batch(s[0]) for s in samps]
                # train decoder
                for s, e in zip(samps, encoded):
                    self.model_decoder.train_on_batch([e, s[1]], np.zeros((e.shape[0], self.z_depth)))

                # test samples
                tsamps = [next(gen) for _ in range(samples)]
                # encode
                tencoded = [self.model_encoder.predict_on_batch(s[0]) for s in tsamps]
                # test decoder
                losses = [self.model_decoder.predict_on_batch([e, s[1]]) for e, s in zip(tencoded, tsamps)]
                values = [rewards_to_values(-loss, discount=self.discount) for loss in losses]
                # train value
                for s, e, v in zip(tsamps, tencoded, values):
                    self.model_value.train_on_batch([s[0], e], v)
            if callback:
                callback(epoch)
