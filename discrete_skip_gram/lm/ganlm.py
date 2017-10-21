import itertools

import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from .lstm_unit import LSTMUnit
from .model import LanguageModel
from ..tensor_util import tensor_one_hot, softmax_nd


class GANLanguageModel(LanguageModel):
    def __init__(self,
                 vocab,
                 initializer,
                 dopt,
                 gopt,
                 hard=True,
                 srng=RandomStreams(123),
                 d_layers=2,
                 d_units=512,
                 g_layers=2,
                 g_units=512,
                 regularizer=None,
                 regularizer_samples=128,
                 regularizer_weight=1e-3,
                 tau0=5,
                 tau_decay=1e-5,
                 tau_min=0.25,
                 constraint=None,
                 eps=1e-9):
        self.eps = T.constant(eps, name='eps', dtype='float32')
        self.vocab = vocab
        self.x_k = len(vocab)
        self.srng = srng
        self.hard=hard
        input_x = T.imatrix(name='input_x')  # (n, depth)
        n = input_x.shape[0]
        depth = input_x.shape[1]

        iteration = K.variable(0, dtype='int32')
        iter_updates = [(iteration, iteration + 1)]
        tau_min = T.constant(tau_min, name='tau_min', dtype='float32')
        tau_decay = T.constant(tau_decay, name='tau_min', dtype='float32')
        tau0 = T.constant(tau0, name='tau_min', dtype='float32')
        tau = tau0 * T.exp(-iteration * tau_decay)
        tau = T.cast(tau, 'float32')
        self.tau = T.nnet.relu(tau - tau_min) + tau_min

        # Discriminator parameters
        self.d_lstms = []
        for i in range(d_layers):
            lstm = LSTMUnit(units=d_units, input_units=[d_units], initializer=initializer)
            self.d_lstms.append(lstm)
        self.d_embed = K.variable(initializer((self.x_k, d_units)), dtype='float32')
        self.yw = K.variable(initializer((d_units, 1)), dtype='float32')
        d_params = list(itertools.chain.from_iterable(l.params for l in self.d_lstms))
        d_params += [self.d_embed, self.yw]

        # Generator parameters
        self.g_lstms = []
        for i in range(g_layers):
            lstm = LSTMUnit(units=g_units, input_units=[g_units], initializer=initializer)
            self.g_lstms.append(lstm)
        self.g_embed = K.variable(initializer((self.x_k + 1, g_units)), dtype='float32')
        self.gw = K.variable(initializer((g_units, self.x_k)), dtype='float32')
        self.gb = K.variable(initializer((self.x_k,)), dtype='float32')
        g_params = list(itertools.chain.from_iterable(l.params for l in self.g_lstms))
        g_params += [self.g_embed, self.gw, self.gb]

        # D(real)
        xreal = T.transpose(input_x, (1, 0))  # (depth, n)
        xreal_one_hot = T.zeros((depth, n, self.x_k))  # (depth, n, x_k)
        mgrid = T.mgrid[0:depth, 0:n]
        xreal_one_hot = T.set_subtensor(xreal_one_hot[mgrid[0], mgrid[1], xreal], 1)  # (depth, n, x_k)
        d_real = self.discriminator(xreal_one_hot)
        d_real = T.mean(d_real, axis=None)

        # Generator
        xfake = self.generator(n=n, depth=depth)  # (depth, n, x_k)
        d_fake = self.discriminator(xfake)
        d_fake = T.mean(d_fake, axis=None)

        # Loss
        d_loss = d_fake - d_real
        g_loss = d_real - d_fake

        # Sampled regularization
        if regularizer_weight > 0 and regularizer_samples > 0:
            idx1 = srng.random_integers(size=(regularizer_samples,), low=0, high=n - 1, dtype='int32')
            idx2 = srng.random_integers(size=(regularizer_samples,), low=0, high=n - 1, dtype='int32')
            s1 = xreal_one_hot[:, idx1, :]  # (depth, samples, x_k)
            s2 = xfake[:, idx2, :]  # (depth, samples, x_k)
            #alpha = srng.uniform(size=(depth, regularizer_samples,), low=0., high=1., dtype='float32').dimshuffle(
            #    (0, 1, 'x')
            #)
            alpha = srng.uniform(size=(regularizer_samples,), low=0., high=1., dtype='float32').dimshuffle(
                ('x', 0, 'x')
            )
            s = (alpha * s1) + ((1. - alpha) * s2) # (depth, samples, x_k)
            d = T.sum(self.discriminator(s)) # scalar
            g = T.grad(d, s) # (depth, samples, x_k)
            g2 = T.sum(T.square(g),axis=(0,2)) # (samples,)
            iwgreg = regularizer_weight * T.mean(g2)
        else:
            iwgreg = T.constant(0.)

        # Parameter regularizers
        d_reg = T.constant(0.)
        if regularizer:
            for p in d_params:
                d_reg += regularizer(p)
        g_reg = T.constant(0.)
        if regularizer:
            for p in g_params:
                g_reg += regularizer(p)

        # Training
        d_total = d_loss + iwgreg + d_reg
        g_total = g_loss + g_reg

        if constraint:
            for p in d_params:
                p.constraint = constraint

        d_updates = dopt.get_updates(d_total, d_params)
        g_updates = gopt.get_updates(g_total, g_params)

        # self.train_d = theano.function([input_x], [], updates=d_updates)
        # self.train_g = theano.function([input_x], [d_loss, g_loss, iwgreg, d_reg, g_reg, tau],
        #                               updates=g_updates + iter_updates)
        # Generation
        input_n = T.iscalar()
        input_depth = T.iscalar()
        xgen = self.generator(n=input_n, depth=input_depth)  # (depth, n, x_k)
        xgen = T.argmax(xgen, axis=2)  # (depth, n)
        xgen = T.transpose(xgen, (1, 0))  # (n, depth)
        self.gen_function = theano.function([input_n, input_depth], xgen)

        # Validation
        # xreal_one_hot: (depth, n, x_k)
        xs = T.concatenate((T.zeros((1, n, self.x_k)), xreal_one_hot[:-1, :, :]), axis=0)  # (depth, n, x_k)
        xs = T.concatenate((T.zeros((depth, n, 1)), xs), axis=2)  # (depth, n, x_k+1)
        xs = T.set_subtensor(xs[0, :, 0], 1)
        y0 = T.dot(xs, self.g_embed)  # (depth, n, units)
        for l in self.g_lstms:
            h1, y1 = l.call([y0])
            y0 = y1
        p = softmax_nd(T.dot(y0, self.gw) + self.gb)  # (depth, n, x_k)
        mgrid = T.mgrid[0:depth, 0:n]
        pt = p[mgrid[0], mgrid[1], xreal]  # (depth, n)
        val_nll = -T.mean(T.log(eps + pt))
        self.val_fun = theano.function([input_x], val_nll)
        self.train_fun = theano.function([input_x], [d_loss, g_loss, iwgreg, d_reg, g_reg, val_nll, tau],
                                         updates=g_updates + iter_updates + d_updates)

        train_headers = ['D Loss', 'G Loss', 'Grad Reg', 'D Reg', 'G Reg', 'NLL', 'Tau']
        val_headers = ['NLL', 'PPL']
        weights = g_params + gopt.weights + d_params + dopt.weights + [iteration]
        super(GANLanguageModel, self).__init__(weights=weights,
                                               train_headers=train_headers,
                                               val_headers=val_headers)

    def grad_l2(self, x):
        # x: (depth, n, x_k)
        d = self.discriminator(x)  # (n,)
        l2s, _ = theano.scan(lambda _i, _d, _x: T.sum(T.square(T.grad(_d[_i], _x)[:, _i, :])),
                             sequences=T.arange(d.shape[0]),
                             outputs_info=[None],
                             non_sequences=[d, x])
        return l2s

    def reg_l2(self, x):
        l2s = self.grad_l2(x)  # (n,)
        return T.mean(T.square(1. - l2s))

    def discriminator(self, xr):
        # xr : (depth, n, x_k)
        y0 = T.dot(xr, self.d_embed)  # (depth, n, units)
        for l in self.d_lstms:
            h1, y1 = l.call([y0])
            y0 = y1
        ylast = y0[-1, :, :]  # (n, units)
        yout = T.dot(ylast, self.yw) #(n, 1)
        return T.flatten(yout) # (n,)
        #yout = T.dot(y0, self.yw)  # (depth, n,1)
        #return T.flatten(yout, ndim=2)  # (depth, n,)

    def generator(self, n, depth):
        rnd = self.srng.uniform(size=(depth, n, self.x_k), low=self.eps, high=1. - self.eps, dtype='float32')
        gumbel = - T.log(self.eps - T.log(self.eps + rnd))  # (depth, n, x_k)
        sequences = [gumbel]
        y0 = T.zeros((n, self.x_k + 1), dtype='float32')
        y0 = T.set_subtensor(y0[:, 0], 1)
        outputs_info = [y0] + [T.repeat(l.h0, repeats=n, axis=0) for l in self.g_lstms]
        non_sequences = list(itertools.chain.from_iterable(l.recurrent_params for l in self.g_lstms))
        non_sequences += [self.g_embed, self.gw, self.gb, self.tau]
        ret, _ = theano.scan(self.generator_scan,
                             sequences=sequences,
                             outputs_info=outputs_info,
                             non_sequences=non_sequences)
        yout = ret[0]  # (depth, n, x_k+1)
        return yout[:, :, 1:]  # (depth, n, x_k)

    def generator_scan(self, gumbel, *params):
        # for i, p in enumerate(params):
        #    print "{}: {}-{}-{}".format(i, p, p.dtype, p.ndim)
        # params
        idx = 0
        # outputs
        y0 = params[idx]  # (n, x_k+1)
        idx += 1
        h0s = params[idx:idx + len(self.g_lstms)]
        idx += len(self.g_lstms)
        # non_sequences
        lstm_params = []
        for l in self.g_lstms:
            p = params[idx:idx + len(l.recurrent_params)]
            idx += len(l.recurrent_params)
            lstm_params.append(p)
        gembed = params[idx]
        idx += 1
        yw = params[idx]
        idx += 1
        yb = params[idx]
        idx += 1
        tau = params[idx]
        idx += 1
        assert idx == len(params)
        # calculations
        h1s = []
        tmp = T.dot(y0, gembed)  # (n, units)
        for l, h0, p in zip(self.g_lstms, h0s, lstm_params):
            h1, y1 = l.step([tmp], h0, p)
            h1s.append(h1)
            tmp = y1
        logits = T.dot(tmp, yw) + yb

        logit_g = (logits + gumbel) / tau  # (n, x_k)
        ysoft = T.nnet.softmax(logit_g)
        if self.hard:
            argmax = T.argmax(logit_g, axis=1)  # (n,)
            yhard = tensor_one_hot(argmax, self.x_k)
            yout = theano.gradient.zero_grad(yhard - ysoft) + ysoft  # (n, x_k)
        else:
            yout = ysoft
        yout = T.concatenate((T.zeros((yout.shape[0], 1)), yout), axis=1)  # (n, x_k+1)
        return [yout] + h1s

    def train_batch(self, xtrain, depth=35, batch_size=64, d_batches=5, **kwargs):
        # for i in range(d_batches):
        #    xsel = self.batch_data(xtrain=xtrain, depth=depth, batch_size=batch_size)
        #    self.train_d(xsel)
        xsel = self.batch_data(xtrain=xtrain, depth=depth, batch_size=batch_size)
        return self.train_fun(xsel)

    def save_output(self, output_path, epoch, xvalid, xtest):
        samples = 64
        depth = 35
        gen = self.gen_function(samples, depth)  # (n, depth)
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
