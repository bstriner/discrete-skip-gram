import csv
import os

import numpy as np
from keras import backend as K
from keras.callbacks import LambdaCallback, CSVLogger
from keras.layers import Input, Embedding, Lambda, Activation
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..layers.encoder_lstm_continuous import EncoderLSTMContinuous
from ..layers.ngram_layer import NgramLayerGenerator
from ..layers.ngram_layer_distributed import NgramLayerDistributed
from ..layers.time_distributed_dense import TimeDistributedDense
from ..layers.utils import drop_dim_2, softmax_nd_layer


class WordSkipgramSequentialSoftmax(object):
    def __init__(self, dataset, units, z_depth, z_k, schedule, window, adversary_weight,
                 frequency=5,
                 kernel_regularizer=None,
                 lr=1e-4, lr_a=3e-4,
                 train_rate=5):
        self.frequency = frequency
        self.train_rate=train_rate
        self.dataset = dataset
        self.units = units
        self.z_depth = z_depth
        self.z_k = z_k
        self.window = window
        self.srng = RandomStreams(123)
        assert (len(schedule.shape) == 1)
        assert (schedule.shape[0] == z_depth)

        x_k = self.dataset.k

        input_x = Input((1,), dtype='int32', name='input_x')
        input_y = Input((window * 2,), dtype='int32', name='input_y')

        # encoder
        embedding = Embedding(x_k, units, embeddings_regularizer=kernel_regularizer)
        embedded_x = drop_dim_2()(embedding(input_x))
        encoder = EncoderLSTMContinuous(z_depth, z_k, units, activation=T.nnet.softmax,
                                        kernel_regularizer=kernel_regularizer)
        z = encoder(embedded_x)  # n, z_depth, z_k

        # decoder
        lstm = LSTM(units, return_sequences=True, kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=kernel_regularizer)
        ngram = NgramLayerDistributed(k=x_k, units=units, kernel_regularizer=kernel_regularizer)
        zh = lstm(z)
        nll = ngram([zh, input_y])

        # loss calculation
        nll_weights = K.variable(schedule, name="schedule", dtype='float32')
        nll_weighted_loss = Lambda(lambda _nll: T.sum(_nll * (nll_weights.dimshuffle(('x', 0))), axis=1, keepdims=True),
                                   output_shape=lambda _nll: (_nll[0], 1))(nll)

        # adversary: try to minimize kl with z
        input_z = Input((self.z_depth, self.z_k), dtype='float32', name='input_z')
        z_shift = Lambda(lambda _z: T.concatenate((T.zeros_like(_z[:, 0:1, :]), _z[:, :-1, :]), axis=1),
                         output_shape=lambda _z: _z)
        alstm = LSTM(units, kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                     return_sequences=True)
        ad1 = TimeDistributedDense(units)
        ad2 = TimeDistributedDense(z_k)
        act = Activation('tanh')
        sm = softmax_nd_layer()
        h = z_shift(input_z)
        h = alstm(h)
        h = ad1(h)
        h = act(h)
        h = ad2(h)
        d = sm(h)
        adversary = Model(inputs=[input_z], outputs=[d])
        print "Adverary model"
        adversary.summary()
        # train adversary
        dz = adversary(z)
        eps = 1e-6
        def kl3d(yt, yp):
            return T.mean(T.sum(T.sum(yt * T.log((yt+eps) / (yp+eps)), axis=2), axis=1), axis=0)

        def em(yt, yp):
            return T.mean(T.sum(T.sum(T.abs_(yt - yp), axis=2), axis=1), axis=0)

        def mse3d(yt, yp):
            return T.mean(T.sum(T.sum(T.square(yt-yp), axis=2), axis=1), axis=0)
        #kl z-dz works ok
        #aloss = mse3d(z, dz)
        #aloss = kl3d(z, dz)
        aloss = em(z, dz)
        aopt = Adam(lr=lr_a)
        # print "A weights: {}".format(adversary.trainable_weights)
        aupdates = aopt.get_updates(adversary.trainable_weights, {}, aloss)
        # self.adversary_train = theano.function([input_x], aloss, updates=aupdates)
        #self.adversary_train = K.function([input_x], aloss, updates=aupdates)
        self.adversary_weight = K.variable(np.float32(adversary_weight), dtype='float32', name='adversary_weight')
        regloss = -self.adversary_weight * aloss
        #regloss = -self.adversary_weight * kl3d(dz, z)

        # adversary_test_z = T.zeros((1,z_depth,z_k), dtype='float32')
        # adversary_test_z._keras_shape = (None,z_depth, z_k)
        # adversary_test_z._uses_learning_phase = False
        # adversary_test_pred = adversary(adversary_test_z)[0,0,:]
        # self.adversary_test = K.function([],adversary_test_pred)
        self.adversary = adversary

        def adversary_loss(ytrue, ypred):
            return aloss

        def loss_f(ytrue, ypred):
            return T.mean(ypred, axis=None)

        def nll_initial(ytrue, ypred):
            return T.mean(nll[:, 0], axis=None)

        def nll_final(ytrue, ypred):
            return T.mean(nll[:, -1], axis=None)

        opt = Adam(lr)
        #adversary.trainable = False
        #for l in adversary.layers:
        #    l.trainable = False
        ngram.add_loss(regloss)
#        ngram.add_update(updates=aupdates)

        self.model = Model(inputs=[input_x, input_y], outputs=[nll_weighted_loss])
        self.model.compile(opt, loss_f, metrics=[nll_initial, nll_final, adversary_loss])
        self.model.summary()
        self.model._make_train_function()

        inputs = self.model._feed_inputs + self.model._feed_targets + self.model._feed_sample_weights
        outputs=[self.model.total_loss] + self.model.metrics_tensors
        trainf = self.model.train_function
        adversary_function = K.function(inputs, outputs, updates=aupdates)

        self.counter = 0
        def train_function(inputs):
            self.counter += 1
            if self.counter > self.train_rate+1:
                self.counter=0
            if self.counter > 0:
                return adversary_function(inputs)
            else:
                return trainf(inputs)

        self.model.train_function = train_function
        # Encoder
        self.model_encode = Model(inputs=[input_x], outputs=[z])

        # Prediction model
        policy = NgramLayerGenerator(ngram, srng=self.srng, depth=self.window * 2)
        zhfinal = Lambda(lambda _z: _z[:, -1, :], output_shape=lambda _z: (_z[0], _z[2]))(zh)
        ypred = policy(zhfinal)
        self.model_predict = Model(inputs=[input_x], outputs=[ypred])

    def write_encodings(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path+".csv", 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word", "Encoding"] + ["P{}".format(j) for j in range(self.z_depth)])
            x = np.arange(self.dataset.k).reshape((-1, 1))
            z = self.model_encode.predict(x, verbose=0)
            for i in range(self.dataset.k):
                enc = z[i, :, :]
                t = np.argmax(enc, axis=1)
                ps = np.max(enc, axis=1)
                encf = "".join(chr(ord('a') + t[j]) for j in range(t.shape[0]))
                psf = [ps[j] for j in range(self.z_depth)]
                word = self.dataset.get_word(i)
                w.writerow([i, word, encf] + psf)
            encodings = np.argmax(z, axis=2)
            np.save(output_path+".npy", encodings)

    def decode_sample(self, x, y):
        word = self.dataset.get_word(x)
        ctx = [self.dataset.get_word(y[i]) for i in range(y.shape[0])]
        lctx = ctx[:self.window]
        rctx = ctx[self.window:]
        return "{} [{}] {}".format(" ".join(lctx), word, " ".join(rctx))

    def write_predictions(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        samples = 8
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Id", "Word"] + ["Sample {}".format(i) for i in range(samples)])
            x = np.arange(self.dataset.k).reshape((-1, 1))
            ys = [self.model_predict.predict(x, verbose=0) for _ in range(samples)]
            for i in range(self.dataset.k):
                word = self.dataset.get_word(i)
                samples = [self.decode_sample(i, y[i, :]) for y in ys]
                w.writerow([i, word] + samples)

    def train(self, batch_size, epochs, steps_per_epoch, output_path, **kwargs):
        gen = self.dataset.skipgram_generator_with_context(n=batch_size, window=self.window)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        def on_epoch_end(epoch, logs):
            if (epoch +1) % self.frequency == 0:
                self.write_encodings(output_path="{}/encoded-{:08d}".format(output_path, epoch))
                self.write_predictions(output_path="{}/predicted-{:08d}.csv".format(output_path, epoch))
                self.model.save_weights("{}/model-{:08d}.h5".format(output_path, epoch))

            #            test = self.adversary_test()
            xtest = np.zeros((1, self.z_depth, self.z_k))
            xtest[:, :, 0] = 1
            test = self.adversary.predict_on_batch(xtest)[0, :, :]
            np.savetxt("{}/test-{}.txt".format(output_path, epoch), test)

            # for i in range(128):
            #    b = next(gen)[0][0]
            #    print "Bshape: {}".format(b.shape)
            #    aloss = self.adversary_train([b])
            #    print "Epoch: {}, Aloss: {}".format(i, aloss)

        csvcb = CSVLogger("{}/history.csv".format(output_path))
        lcb = LambdaCallback(on_epoch_begin=on_epoch_end)
        cbs = [csvcb, lcb]
        self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=cbs, **kwargs)
