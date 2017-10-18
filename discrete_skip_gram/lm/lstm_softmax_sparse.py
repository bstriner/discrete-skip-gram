import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from .lstm_unit import LSTMUnit
from .model import LanguageModel
from ..tensor_util import softmax_nd
from ..util import generate_batch_indices


class LSTMSoftmaxSparse(LanguageModel):
    def __init__(self,
                 vocab,
                 coding,
                 units,
                 opt,
                 initializer,
                 srng,
                 zoneout=0.5,
                 input_droput=0.5,
                 output_dropout=0.5,
                 eps=1e-9):

        # Parameters
        self.lstm = LSTMUnit(
            input_units=[units],
            units=units,
            initializer=initializer
        )
        self.vocab = vocab
        self.coding= theano.shared(np.int32(coding))
        assert len(coding.shape) == 2
        code_k = coding.shape[1]
        x_k = len(vocab)
        xembed = K.variable(initializer((x_k + 1, units)))
        yw = K.variable(initializer((units, code_k)))
        yb = K.variable(initializer((code_k,)))
        self.params = [xembed, yw, yb] + self.lstm.params

        # Input
        input_x = T.imatrix(name='input_x')  # (n, depth)
        n = input_x.shape[0]
        depth = input_x.shape[1]

        # Training

        xr = T.transpose(input_x, (1, 0))  # (depth, n)
        xrs = T.concatenate((T.zeros((1, n), dtype='int32'), xr[:-1, :] + 1), axis=0)

        xembedded = xembed[xrs, :]
        if input_droput > 0:
            input_dropout_mask = T.cast(srng.binomial(size=(depth, n, units), p=input_droput, n=1), 'float32')
            xembedded = (xembedded * input_dropout_mask) / input_droput

        zoneout_mask = T.cast(srng.binomial(size=(depth, n, units), p=zoneout, n=1), 'float32')
        sequences = [xembedded, zoneout_mask]
        # outputs_info = [self.lstm.h0]
        outputs_info = [T.repeat(self.lstm.h0, repeats=n, axis=0), None]
        non_sequences = self.lstm.recurrent_params
        (h1, y1), _ = theano.scan(self.scan,
                                  sequences=sequences,
                                  outputs_info=outputs_info,
                                  non_sequences=non_sequences)

        if output_dropout > 0:
            output_dropout_mask = T.cast(srng.binomial(size=(depth, n, units), p=output_dropout, n=1), 'float32')
            y1 = (y1 * output_dropout_mask) / output_dropout

        p1 = T.nnet.sigmoid(T.dot(y1, yw)+yb) # (depth, n, code)
        xcode = coding[xr, :] # (depth, n, code)
        p = (xcode*p1) + ((1-xcode)*(1-p1))
        nllr = T.sum(T.log2(eps+p), axis=2) # (depth, n)
        nll = T.mean(nllr, axis=None)

        updates = opt.get_updates(nll, self.params)
        self.train_fun = theano.function([input_x], [nll], updates=updates)

        # Validation
        xembedded = xembed[xrs, :]
        sequences = [xembedded]
        (h1, y1), _ = theano.scan(self.scan_val,
                                  sequences=sequences,
                                  outputs_info=outputs_info,
                                  non_sequences=non_sequences)
        p1 = T.nnet.sigmoid(T.dot(y1, yw) + yb)  # (depth, n, code)
        p = (xcode*p1) + ((1-xcode)*(1-p1))
        nllr = T.sum(T.log2(eps+p), axis=2) # (depth, n)
        nll_part = T.transpose(nllr, (1,0)) # (n, depth)
        self.nll_fun = theano.function([input_x], nll_part)

        # Generation
        gen_n = T.iscalar(name='n')
        gen_depth = T.iscalar(name='depth')
        rnd = srng.uniform(low=0., high=1., dtype='float32', size=(gen_depth, gen_n))
        sequences = [rnd]
        outputs_info = [T.repeat(self.lstm.h0, repeats=gen_n, axis=0), T.zeros((gen_n,), dtype='int32')]
        non_sequences = [xembed, yw, yb] + self.lstm.recurrent_params
        (h1, x1r), _ = theano.scan(self.scan_gen,
                                   sequences=sequences,
                                   outputs_info=outputs_info,
                                   non_sequences=non_sequences)
        x1 = T.transpose(x1r, (1, 0)) - 1
        self.gen_fun = theano.function([gen_n, gen_depth], x1)

        train_headers = ['NLL']
        val_headers = ['NLL', 'PPL']
        weights = self.params + opt.weights
        super(LSTMSoftmaxSparse, self).__init__(weights=weights,
                                                 train_headers=train_headers,
                                                 val_headers=val_headers)

    def scan(self, x0, zo, h0, *params):
        assert h0.ndim == 2
        h1, y1 = self.lstm.step(xs=[x0], h0=h0, params=params)
        h1 = (zo * h0) + ((1. - zo) * h1)  # zoneout
        return [h1, y1]

    def scan_val(self, x0, h0, *params):
        assert h0.ndim == 2
        h1, y1 = self.lstm.step(xs=[x0], h0=h0, params=params)
        return [h1, y1]

    def scan_gen(self, rng, h0, x0, xembed, yw, yb, *params):
        assert h0.ndim == 2
        xe = xembed[x0, :]
        h1, y1 = self.lstm.step(xs=[xe], h0=h0, params=params)
        p1 = softmax_nd(T.dot(y1, yw) + yb)
        cs = T.cumsum(p1, axis=1)
        x1 = T.sum(T.gt(rng.dimshuffle((0, 'x')), cs), axis=1)
        x1 = T.clip(x1, 0, cs.shape[1] - 1)
        x1 = T.cast(x1 + 1, 'int32')
        return [h1, x1]

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
        return [np.asscalar(avgnll), np.asscalar(np.power(2, avgnll))]

    def train_batchx(self, x, **kwargs):
        return self.train_fun(x)
