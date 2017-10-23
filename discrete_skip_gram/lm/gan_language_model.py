import itertools

import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from .lstm_unit import LSTMUnit
from .model import LanguageModel
from ..tensor_util import softmax_nd


class GANModel(LanguageModel):
    def __init__(self,
                 vocab,
                 initializer,
                 dopt,
                 gopt,
                 hard=True,
                 srng=RandomStreams(123),
                 d_layers=2,
                 d_units=512,
                 d_input_dropout=0.1,
                 d_zoneout=0.5,
                 d_dropout=0.5,
                 g_layers=2,
                 g_units=1024,
                 g_input_dropout=0.1,
                 g_zoneout=0.5,
                 g_dropout=0.5,
                 d_regularizer=None,
                 g_regularizer=None,
                 regularizer_weight=1e-3,
                 constraint=None,
                 eps=1e-9):
        self.eps = T.constant(eps, name='eps', dtype='float32')
        self.vocab = vocab
        self.x_k = len(vocab)
        self.srng = srng
        self.hard = hard
        self.d_units = d_units
        self.g_units = g_units
        self.d_dropout = d_dropout
        self.d_input_dropout = d_input_dropout
        self.g_input_dropout = g_input_dropout
        self.g_zoneout = g_zoneout
        self.g_dropout = g_dropout
        # self.g_dropout = g_dropout
        # self.g_input_dropout = g_input_dropout
        # self.g_zoneout = g_zoneout
        input_x = T.imatrix(name='input_x')  # (n, depth)
        n = input_x.shape[0]
        depth = input_x.shape[1]

        # Discriminator parameters
        self.d_lstms = []
        for i in range(d_layers):
            lstm = LSTMUnit(units=d_units,
                            input_units=[d_units],
                            initializer=initializer,
                            srng=srng,
                            zoneout=d_zoneout)
            self.d_lstms.append(lstm)
        self.d_embed = K.variable(initializer((self.x_k + 1, d_units)), dtype='float32')
        self.dw = K.variable(initializer((d_units, self.x_k)), dtype='float32')
        self.db = K.variable(initializer((self.x_k,)), dtype='float32')
        d_params = list(itertools.chain.from_iterable(l.params for l in self.d_lstms))
        d_params += [self.d_embed, self.dw, self.db]

        # Generator parameters
        self.g_lstms = []
        for i in range(g_layers):
            lstm = LSTMUnit(units=g_units,
                            input_units=[g_units],
                            initializer=initializer,
                            zoneout=g_zoneout,
                            srng=srng)
            self.g_lstms.append(lstm)
        self.g_embed = K.variable(initializer((self.x_k + 1, g_units)), dtype='float32')
        self.gw = K.variable(initializer((g_units, self.x_k)), dtype='float32')
        self.gb = K.variable(initializer((self.x_k,)), dtype='float32')
        g_params = list(itertools.chain.from_iterable(l.params for l in self.g_lstms))
        g_params += [self.g_embed, self.gw, self.gb]

        # D(real)
        xreal = T.transpose(input_x, (1, 0))  # (depth, n)
        xreal_shifted = T.concatenate((T.zeros((1, n), dtype='int32'), xreal[:-1, :] + 1), axis=0)  # (depth, n)
        yreal = self.discriminator(xreal_shifted)  # (depth, n, x_k)
        mgrid = T.mgrid[0:depth, 0:n]
        yreal_t = yreal[mgrid[0], mgrid[1], xreal]  # (depth, n)
        yrealmean = T.mean(yreal_t, axis=None)  # scalar

        # Generator
        gen_p, gen_x = self.generator(n=n, depth=depth)  # (depth, n, x_k), (depth, n)
        gen_x_shifted = T.concatenate((T.zeros((1, n), dtype='int32'), gen_x[:-1, :] + 1), axis=0)  # (depth, n)

        # D(fake)
        yfake = self.discriminator(gen_x_shifted)  # (depth, n, x_k)
        yfake_t = T.sum(yfake * gen_p, axis=2)  # (depth, n)
        yfakemean = T.mean(yfake_t, axis=None)  # scalar

        # Loss
        d_loss = yfakemean - yrealmean
        g_loss = -yfakemean

        # Regularization
        mode = 1
        if mode == 0:
            yfakemin = T.min(yfake, axis=2)  # (depth, n)
            yrealmin = T.min(yreal, axis=2)  # (depth, n)
            reg1 = T.mean(T.square(1. + yfakemin))
            reg2 = T.mean(T.square(1. + yrealmin))
            wganreg = regularizer_weight * (reg1 + reg2)
        else:
            reg1 = T.mean(T.square(T.nnet.relu(-1 - yfake)))
            reg2 = T.mean(T.square(T.nnet.relu(-1 - yreal)))
            wganreg = regularizer_weight * (reg1 + reg2)

        d_reg = T.constant(0.)
        if d_regularizer:
            for p in d_params:
                d_reg += d_regularizer(p)
        g_reg = T.constant(0.)
        if g_regularizer:
            for p in g_params:
                g_reg += g_regularizer(p)

        # NLL Calculation
        xe = self.g_embed[xreal_shifted, :]  # (depth, n, units)
        tmp = xe
        for l in self.g_lstms:
            h1, y1 = l.call([tmp], val=True)
            tmp = y1
        p1 = softmax_nd(T.dot(tmp, self.gw) + self.gb)  # (depth, n, x_k)
        p1_t = p1[mgrid[0], mgrid[1], xreal]  # (depth, n)
        nll_t = -T.log(self.eps + p1_t)  # (depth, n)
        nll = T.mean(nll_t)

        # training
        d_total = d_loss + wganreg + d_reg
        g_total = g_loss + g_reg
        d_updates = dopt.get_updates(d_total, d_params)
        g_updates = gopt.get_updates(g_total, g_params)
        self.train_fun = theano.function([input_x], [d_loss, g_loss, wganreg, d_reg, g_reg, nll],
                                         updates=g_updates + d_updates)
        self.train_fun_d = theano.function([input_x], [],
                                           updates=d_updates)
        self.train_fun_g = theano.function([input_x], [d_loss, g_loss, wganreg, d_reg, g_reg, nll],
                                           updates=g_updates)

        # validation
        self.val_fun = theano.function([input_x], nll)
        # Generation
        input_n = T.iscalar()
        input_depth = T.iscalar()
        gen_p, gen_xr = self.generator(n=input_n, depth=input_depth)
        gen_x = T.transpose(gen_xr, (1, 0))  # (n, depth)
        self.gen_fun = theano.function([input_n, input_depth], gen_x)

        train_headers = ['D Loss', 'G Loss', 'Grad Reg', 'D Reg', 'G Reg', 'NLL']
        val_headers = ['NLL', 'PPL']
        weights = g_params + gopt.weights + d_params + dopt.weights
        super(GANModel, self).__init__(weights=weights,
                                       train_headers=train_headers,
                                       val_headers=val_headers)

    def train_batchx(self, x, **kwargs):
        return self.train_fun(x)

    def discriminator(self, x):
        # x: (depth, n) shifted+1
        xe = self.d_embed[x, :]  # (depth, n, units)
        if self.d_input_dropout > 0:
            mask = self.srng.binomial(size=(xe.shape[1], xe.shape[2]), n=1, p=1. - self.d_input_dropout,
                                      dtype='float32').dimshuffle(('x', 0, 1))
            xe = (mask * xe) / (1. - self.d_input_dropout)
        tmp = xe
        for l in self.d_lstms:
            h1, y1 = l.call([tmp])
            if self.d_dropout > 0:
                mask = self.srng.binomial(size=(y1.shape[1], y1.shape[2]), n=1, p=1. - self.d_dropout,
                                          dtype='float32').dimshuffle(('x', 0, 1))
                y1 = (mask * y1) / (1. - self.d_dropout)
            tmp = y1
        tmp = T.dot(tmp, self.dw) + self.db  # (depth, n, x_k)
        tmp -= T.max(tmp, axis=2, keepdims=True)
        return tmp

    def train_batch(self, xtrain, depth=35, batch_size=64, d_batches=5, **kwargs):
        mode = 1
        if mode == 0:
            for _ in range(d_batches):
                xsel = self.batch_data(xtrain=xtrain, depth=depth, batch_size=batch_size)
                self.train_fun_d(xsel)
            xsel = self.batch_data(xtrain=xtrain, depth=depth, batch_size=batch_size)
            return self.train_fun_g(xsel)
        else:
            xsel = self.batch_data(xtrain=xtrain, depth=depth, batch_size=batch_size)
            return self.train_fun(xsel)

    def generator(self, n, depth):
        rnd = self.srng.uniform(size=(depth, n, self.x_k), low=self.eps, high=1. - self.eps, dtype='float32')
        gumbel = - T.log(self.eps - T.log(self.eps + rnd))  # (depth, n, x_k)
        sequences = [gumbel]
        if self.g_zoneout > 0:
            for _ in self.g_lstms:
                sequences.append(self.srng.binomial(size=(depth, n, self.g_units), n=1, p=self.g_zoneout,
                                                    dtype='float32'))
        x0 = T.zeros((n,), dtype='int32')
        outputs_info = [None, x0] + [T.repeat(l.h0, repeats=n, axis=0) for l in self.g_lstms]
        non_sequences = list(itertools.chain.from_iterable(l.recurrent_params for l in self.g_lstms))
        non_sequences += [self.g_embed, self.gw, self.gb]
        if self.g_input_dropout > 0:
            non_sequences.append(self.srng.binomial(size=(n, self.g_units), n=1, p=1. - self.g_input_dropout,
                                                    dtype='float32'))
        if self.g_dropout > 0:
            for _ in self.g_lstms:
                non_sequences.append(self.srng.binomial(size=(n, self.g_units), n=1, p=1. - self.g_dropout,
                                                        dtype='float32'))
        ret, _ = theano.scan(self.generator_scan,
                             sequences=sequences,
                             outputs_info=outputs_info,
                             non_sequences=non_sequences)
        p1r = ret[0]  # (depth, n, x_k)
        x1r = ret[1] - 1  # (depth, n)
        return p1r, x1r

    def generator_scan(self, gumbel, *params):
        idx = 0
        if self.g_zoneout > 0:
            zos = params[idx:idx + len(self.g_lstms)]
            idx += len(self.g_lstms)
        # outputs
        x0 = params[idx]
        idx += 1
        h0s = params[idx:idx + len(self.g_lstms)]
        idx += len(self.g_lstms)
        # non_sequences
        lstm_params = []
        for l in self.g_lstms:
            lstm_params.append(params[idx:idx + len(l.recurrent_params)])
            idx += len(l.recurrent_params)
        gembed = params[idx]
        idx += 1
        gw = params[idx]
        idx += 1
        gb = params[idx]
        idx += 1
        if self.g_input_dropout > 0:
            ido = params[idx]
            idx += 1
        if self.g_dropout > 0:
            dos = params[idx:idx + len(self.g_lstms)]
            idx += len(self.g_lstms)
        assert idx == len(params)

        tmp = gembed[x0, :]  # (n, units)
        if self.g_input_dropout > 0:
            tmp = (tmp * ido) / (1. - self.g_input_dropout)
        h1s = []
        for i, (l, h0, p) in enumerate(zip(self.g_lstms, h0s, lstm_params)):
            h1, tmp2 = l.step([tmp], h0, p)
            if self.g_zoneout > 0:
                zo = zos[i]
                h1 = (zo * h0) + ((1. - zo) * h1)
            if self.g_dropout > 0:
                do = dos[i]
                tmp2 = (tmp2 * do) / (1. - self.g_dropout)
            h1s.append(h1)
            tmp = tmp2
        logits = T.dot(tmp, gw) + gb  # (n, x_k)
        p1 = T.nnet.softmax(logits)
        x1 = T.argmax(logits + gumbel, axis=1) + 1  # (n,)
        x1 = T.cast(x1, 'int32')
        return [p1, x1] + h1s

    def save_output(self, output_path, epoch, xvalid, xtest):
        samples = 64
        depth = 20
        gen = self.gen_fun(samples, depth)  # (n, depth)
        assert gen.shape[0] == samples
        assert gen.shape[1] == depth
        with open('{}/generated-{:08d}.txt'.format(output_path, epoch), 'w') as f:
            for i in range(gen.shape[0]):
                s = " ".join(self.vocab[j] for j in gen[i, :])
                f.write(s + "\n")

    def validate(self, x, batch_size=64, depth=35, val_batches=1000, **kwargs):
        nlls = []
        for _ in tqdm(range(val_batches), desc='Validation'):
            xsel = self.batch_data(xtrain=x, depth=depth, batch_size=batch_size)
            nll = self.val_fun(xsel)
            nlls.append(nll)
        avgnll = np.mean(nlls)
        return [np.asscalar(avgnll), np.asscalar(np.power(2, avgnll))]
