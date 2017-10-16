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
                 units,
                 x_k,
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
        xembed = K.variable(initializer((x_k + 1, units)))
        yw = K.variable(initializer((units, x_k)))
        yb = K.variable(initializer((x_k,)))
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
        # y1: (depth, n, units)
        p1 = softmax_nd(T.dot(y1, yw) + yb)  # (depth, n, x_k)
        # p1: (depth, n, x_k)
        mgrid = T.mgrid[0:depth, 0:n]
        pt = p1[mgrid[0], mgrid[1], xr]  # (depth, n)
        nllr = -T.log2(eps + pt)  # (depth, n)
        nll = T.mean(nllr, axis=None)  # scalar
        # ppl = T.mean(T.power(2, logt), axis=None) # scalar
        updates = opt.get_updates(nll, self.params)
        self.train_fun = theano.function([input_x], [nll], updates=updates)

        # Validation
        xembedded = xembed[xrs, :]
        sequences = [xembedded]
        (h1, y1), _ = theano.scan(self.scan_val,
                                  sequences=sequences,
                                  outputs_info=outputs_info,
                                  non_sequences=non_sequences)
        p1 = softmax_nd(T.dot(y1, yw) + yb)  # (depth, n, x_k)
        mgrid = T.mgrid[0:depth, 0:n]
        pt = p1[mgrid[0], mgrid[1], xr]  # (depth, n)
        nllr = -T.log2(eps + pt)  # (depth, n)
        nll = T.transpose(nllr, (1, 0))
        # avg_ppl = T.mean(ppl, axis=None)

        train_headers = ['NLL']
        val_headers = ['NLL', 'PPL']
        weights = self.params + opt.weights
        self.nll_fun = theano.function([input_x], nll)
        super(LSTMSoftmaxVanilla, self).__init__(weights=weights,
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

    def save_output(self, epoch, xvalid, xtest):
        pass

    def validate(self, x, batch_size=64, depth=35, **kwargs):
        # calc perplexity on test set
        stack = []
        n = x.shape[0]
        idx = list(generate_batch_indices(n=n-depth+1, batch_size=batch_size))
        for idx0, idx1 in tqdm(idx, desc='Validating'):
            i1 = np.arange(idx0, idx1).reshape((-1, 1))
            i2 = np.arange(depth).reshape((1, -1))
            i = i1 + i2
            xb = x[i]
            nll = self.nll_fun(xb)
            stack.append(nll)
        nll = np.concatenate(stack, axis=0)
        p0 = nll[0, :]  # (d,)
        p1 = nll[1:, depth-1]  # (n-d,)
        nllsel = np.concatenate((p0, p1), axis=0)
        assert nllsel.shape[0] == x.shape[0]
        avgnll = np.mean(nllsel)
        return [np.asscalar(avgnll), np.asscalar(np.power(2, avgnll))]

    def train_batchx(self, x, **kwargs):
        return self.train_fun(x)
