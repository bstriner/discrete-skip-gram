import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from .lstm_unit import LSTMUnit
from .model import LanguageModel
from ..tensor_util import softmax_nd
from ..util import generate_batch_indices


class LSTMSoftmaxVanilla(LanguageModel):
    def __init__(self,
                 vocab,
                 units,
                 opt,
                 initializer,
                 srng,
                 layers=1,
                 regularizer=None,
                 activity_reg=0,
                 temporal_activity_reg=0,
                 zoneout=0.5,
                 input_droput=0.1,
                 output_dropout=0.5,
                 eps=1e-9):
        assert layers > 0
        # Parameters
        self.vocab = vocab
        self.zoneout = zoneout
        x_k = len(vocab)
        xembed = K.variable(initializer((x_k + 1, units)))
        yw = K.variable(initializer((units, x_k)))
        yb = K.variable(initializer((x_k,)))
        self.lstms = []
        self.params = [xembed, yw, yb]
        for i in range(layers):
            lstm = LSTMUnit(
                input_units=[units],
                units=units,
                initializer=initializer
            )
            self.lstms.append(lstm)
            self.params += lstm.params

        # Input
        input_x = T.imatrix(name='input_x')  # (n, depth)
        n = input_x.shape[0]
        depth = input_x.shape[1]

        # Training

        xr = T.transpose(input_x, (1, 0))  # (depth, n)
        xrs = T.concatenate((T.zeros((1, n), dtype='int32'), xr[:-1, :] + 1), axis=0)

        xembedded = xembed[xrs, :]
        if input_droput > 0:
            input_dropout_mask = T.cast(srng.binomial(size=(n, units), p=1. - input_droput, n=1),
                                        'float32').dimshuffle(('x', 0, 1))
            xembedded = (xembedded * input_dropout_mask) / (1. - input_droput)

        y0 = xembedded
        y1s = []
        for i in range(layers):
            lstm = self.lstms[i]
            zoneout_mask = T.cast(srng.binomial(size=(depth, n, units), p=zoneout, n=1), 'float32')
            sequences = [y0, zoneout_mask]
            outputs_info = [T.repeat(lstm.h0, repeats=n, axis=0), None]
            non_sequences = lstm.recurrent_params
            (h1, y1), _ = theano.scan(self.scan(i),
                                      sequences=sequences,
                                      outputs_info=outputs_info,
                                      non_sequences=non_sequences)
            y1s.append(y1)
            if output_dropout > 0:
                output_dropout_mask = T.cast(srng.binomial(size=(n, units), p=1. - output_dropout, n=1),
                                             'float32').dimshuffle(('x', 0, 1))
                y1 = (y1 * output_dropout_mask) / (1. - output_dropout)
            y0 = y1
        # y1: (depth, n, units)
        p1 = softmax_nd(T.dot(y0, yw) + yb)  # (depth, n, x_k)
        # p1: (depth, n, x_k)
        mgrid = T.mgrid[0:depth, 0:n]
        pt = p1[mgrid[0], mgrid[1], xr]  # (depth, n)
        nllr = -T.log(eps + pt)  # (depth, n)
        nll = T.mean(nllr, axis=None)  # scalar
        # ppl = T.mean(T.power(2, logt), axis=None) # scalar

        loss_activity = T.constant(0.)
        loss_temporal_activity = T.constant(0.)
        loss_param_reg = T.constant(0.)
        if activity_reg > 0:
            for h in y1s:
                loss_activity += activity_reg * T.mean(T.square(h), axis=None)
        if temporal_activity_reg > 0:
            for h in y1s:
                loss_temporal_activity += temporal_activity_reg * T.mean(T.square((h[1:, :, :]) - (h[:-1, :, :])),
                                                                         axis=None)
        if regularizer:
            for p in self.params:
                if p.ndim > 1:
                    loss_param_reg += regularizer(p)
        loss = nll + loss_activity + loss_temporal_activity + loss_param_reg

        updates = opt.get_updates(loss, self.params)
        self.train_fun = theano.function([input_x], [nll, loss_activity, loss_temporal_activity, loss_param_reg, loss],
                                         updates=updates)

        # Validation
        xembedded = xembed[xrs, :]
        y0 = xembedded
        for i in range(layers):
            lstm = self.lstms[i]
            sequences = [y0]
            outputs_info = [T.repeat(lstm.h0, repeats=n, axis=0), None]
            non_sequences = lstm.recurrent_params
            (h1, y1), _ = theano.scan(self.scan_val(i),
                                      sequences=sequences,
                                      outputs_info=outputs_info,
                                      non_sequences=non_sequences)
            y0 = y1
        p1 = softmax_nd(T.dot(y0, yw) + yb)  # (depth, n, x_k)
        mgrid = T.mgrid[0:depth, 0:n]
        pt = p1[mgrid[0], mgrid[1], xr]  # (depth, n)
        nllr = -T.log(eps + pt)  # (depth, n)
        nll = T.transpose(nllr, (1, 0))
        self.nll_fun = theano.function([input_x], nll)

        # Generation
        gen_n = T.iscalar(name='n')
        gen_depth = T.iscalar(name='depth')
        rnd = srng.uniform(low=0., high=1., dtype='float32', size=(gen_depth, gen_n))
        sequences = [rnd]
        outputs_info = [T.zeros((gen_n,), dtype='int32')]
        for lstm in self.lstms:
            outputs_info.append(T.repeat(lstm.h0, repeats=gen_n, axis=0))
        non_sequences = [xembed, yw, yb]
        for lstm in self.lstms:
            non_sequences += lstm.recurrent_params
        ret, _ = theano.scan(self.scan_gen,
                             sequences=sequences,
                             outputs_info=outputs_info,
                             non_sequences=non_sequences)
        x1r = ret[0]
        x1 = T.transpose(x1r, (1, 0)) - 1
        self.gen_fun = theano.function([gen_n, gen_depth], x1)

        train_headers = ['NLL', 'Activity Reg', 'Temporal Reg', 'Weight Reg', 'Loss']
        val_headers = ['NLL', 'PPL']
        weights = self.params + opt.weights
        super(LSTMSoftmaxVanilla, self).__init__(weights=weights,
                                                 train_headers=train_headers,
                                                 val_headers=val_headers)

    def scan(self, i):
        def fun(x0,
                zo,
                h0, *params):
            assert h0.ndim == 2
            h1, y1 = self.lstms[i].step(xs=[x0], h0=h0, params=params)
            h1 = (zo * h0) + ((1. - zo) * h1)  # zoneout
            return [h1, y1]

        return fun

    def scan_val(self, i):
        def fun(x0, h0, *params):
            assert h0.ndim == 2
            h1, y1 = self.lstms[i].step(xs=[x0], h0=h0, params=params)
            h1 = (self.zoneout * h0) + ((1 - self.zoneout) * h1)
            return [h1, y1]

        return fun

    def scan_gen(self, rng, x0, *params):
        # sequences, outputs, non_sequences
        idx = 0
        h0s = params[idx:idx + len(self.lstms)]
        idx += len(self.lstms)

        xembed = params[idx]
        idx += 1
        yw = params[idx]
        idx += 1
        yb = params[idx]
        idx += 1

        xe = xembed[x0, :]
        y0 = xe
        h1s = []
        for i, lstm in enumerate(self.lstms):
            p = params[idx:idx + len(lstm.recurrent_params)]
            idx += len(lstm.recurrent_params)
            h1, y1 = lstm.step(xs=[y0], h0=h0s[i], params=p)
            h1s.append(h1)
            y0 = y1
        p1 = softmax_nd(T.dot(y0, yw) + yb)
        cs = T.cumsum(p1, axis=1)
        x1 = T.sum(T.gt(rng.dimshuffle((0, 'x')), cs), axis=1)
        x1 = T.clip(x1, 0, cs.shape[1] - 1)
        x1 = T.cast(x1 + 1, 'int32')
        assert idx == len(params)
        return [x1] + h1s

    def save_output(self, output_path, epoch, xvalid, xtest):
        samples = 64
        depth = 35
        x = self.gen_fun(samples, depth)
        with open('{}/generated-{:08d}.txt'.format(output_path, epoch), 'w') as f:
            for i in range(x.shape[0]):
                s = []
                for j in range(x.shape[1]):
                    s.append(self.vocab[x[i, j]])
                f.write(" ".join(s) + "\n")

    def validate(self, x, batch_size=64, depth=35, **kwargs):
        # calc perplexity on test set
        stack = []
        n = x.shape[0]
        idx = list(generate_batch_indices(n=n - depth + 1, batch_size=batch_size))
        for idx0, idx1 in tqdm(idx, desc='Validating'):
            i1 = np.arange(idx0, idx1).reshape((-1, 1))
            i2 = np.arange(depth).reshape((1, -1))
            i = i1 + i2
            xb = x[i]
            nll = self.nll_fun(xb)
            stack.append(nll)
        nll = np.concatenate(stack, axis=0)
        p0 = nll[0, :]  # (d,)
        p1 = nll[1:, depth - 1]  # (n-d,)
        nllsel = np.concatenate((p0, p1), axis=0)
        assert nllsel.shape[0] == x.shape[0]
        avgnll = np.mean(nllsel)
        return [np.asscalar(avgnll), np.asscalar(np.exp(avgnll))]

    def train_batchx(self, x, **kwargs):
        return self.train_fun(x)
