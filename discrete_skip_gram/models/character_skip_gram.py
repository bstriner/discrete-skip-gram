from ..layers.encoder_lstm import EncoderLSTM
from ..layers.character_lstm import CharacterLSTM
from ..layers.discrete_lstm import DiscreteLSTM
from ..layers.decoder_merge import DecoderMerge
from keras.layers import Input, Lambda
from theano.tensor.shared_randomstreams import RandomStreams
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

import theano.tensor as T






def loss_layer(y_k):
    def f(_x):
        _dec_y, _y, _enc_p = _x
        # build one-hot targets
        target_y = T.zeros((_y.shape[0], _y.shape[1], y_k + 2), dtype='int32')
        ind1 = T.repeat(T.arange(_y.shape[0], dtype='int32').reshape((-1, 1)), _y.shape[1], axis=1)
        ind2 = T.repeat(T.arange(_y.shape[1], dtype='int32').reshape((1, -1)), _y.shape[0], axis=0)
        target_y = T.set_subtensor(target_y[T.flatten(ind1), T.flatten(ind2), T.flatten(_y)], 1)
        target_y = target_y.dimshuffle((0, 'x', 1, 'x', 2))
        # calculate loss
        # _dec_y = T.clip(_dec_y, K.epsilon(), 1.0 - K.epsilon())
        loss_raw = (-target_y * T.log(_dec_y)) #+ (-(1 - target_y) * T.log(1 - _dec_y))
        loss_weighted = loss_raw * (_enc_p.dimshuffle((0, 1, 'x', 2, 'x')))
        # mask loss
        mask = T.neq(_y, 0).dimshuffle((0, 'x', 1, 'x', 'x'))
        loss = mask * loss_weighted
        return T.sum(T.sum(T.sum(T.sum(loss, axis=-1), axis=-1), axis=-1), axis=-1, keepdims=True)

    return Lambda(f, output_shape=lambda _x: (_x[0][0], 1), name="loss_layer")


class CharacterSkipGram(object):
    def __init__(self, latent_depth, latent_k, x_k, y_k, units):
        print("latent_k {}, x_k {}, y_k {}".format(latent_k, x_k, y_k))
        char_lstm = CharacterLSTM(x_k, units=units)
        encoder_lstm = EncoderLSTM(latent_k, units=units)
        decoder_h_lstm = DiscreteLSTM(latent_k + 1, units=units, return_sequences=True)
        decoder_hy_lstm = DiscreteLSTM(y_k + 3, units=units, return_sequences=True)
        decoder_merge = DecoderMerge(y_k=y_k, z_k=latent_k, units=units)

        x = Input((None,), dtype='int32')
        y = Input((None,), dtype='int32')

        srng = RandomStreams(123)
        rng = Lambda(lambda _x: srng.uniform(size=(_x.shape[0], latent_depth), low=0, high=1, dtype='float32'),
                     output_shape=lambda _x: (x[0], latent_depth))(x)

        enc_h = char_lstm(x)
        enc_p, enc_z = encoder_lstm([enc_h, rng])
        y_shifted = shift_tensor()(y)
        enc_z_shifted = shift_tensor()(enc_z)
        dec_h = decoder_h_lstm(enc_z_shifted)
        dec_hy = decoder_hy_lstm(y_shifted)

        # m1 = Model(inputs=[x, y], outputs=[dec_hy, dec_h])
        # l = m1.predict([np.random.randint(0, x_k, (32, 6)), np.random.randint(0, y_k, (32, 7))])
        # print "L1: {}, {}".format(l[0].shape, l[1].shape)

        dec_y = decoder_merge([dec_hy, dec_h])
        # m1 = Model(inputs=[x, y], outputs=[dec_y, y, enc_p])
        # m1.summary()
        # l = m1.predict([np.random.randint(0, x_k, (32, 6)), np.random.randint(0, y_k, (32, 7))])
        # print "dec_y {}, y {}, enc_p {}".format(l[0].shape, l[1].shape, l[2].shape)

        loss = loss_layer(y_k)([dec_y, y, enc_p])
        self.model = Model(inputs=[x, y], outputs=[loss])
        # self.model.summary()
        # l = self.model.predict([np.random.randint(0, x_k, (32, 6)), np.random.randint(0, y_k, (32, 7))])
        # print "L3: {}".format(l.shape)
        # self.model.add_loss(loss_raw)
        self.model.compile(Adam(1e-3), [custom_loss])

        self.encoder = Model(inputs=[x], outputs=[enc_z])
