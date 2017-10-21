import keras.backend as K
import numpy as np
import theano
import theano.tensor as T

from .lstm_model import LSTMModel
from .model import LanguageModel
from ..tensor_util import softmax_nd


class LSTMSoftmaxSparse(LanguageModel):
    def __init__(self,
                 vocab,
                 encoding,
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
        # Parameters
        self.vocab = vocab
        self.encoding = T.constant(np.int32(encoding), name='encoding')
        assert len(encoding.shape) == 2
        x_k = encoding.shape[0]
        code_k = encoding.shape[1]
        self.lstm = LSTMModel(
            x_k=x_k,
            srng=srng,
            initializer=initializer,
            units=units,
            layers=layers,
            activity_reg=activity_reg,
            temporal_activity_reg=temporal_activity_reg,
            zoneout=zoneout,
            input_droput=input_droput,
            output_dropout=output_dropout
        )

        yw = K.variable(initializer((units, code_k)))
        yb = K.variable(initializer((code_k,)))
        self.params = self.lstm.params + [yw, yb]

        # Training
        p1 = T.nnet.sigmoid(T.dot(self.lstm.train_y, yw) + yb)  # (depth, n, code)
        xcode = self.encoding[self.lstm.xr, :]  # (depth, n, code)
        assert xcode.ndim == 3
        # nllrp = (xcode * T.log2(eps + p1)) + ((1 - xcode) * (T.log2(eps + 1. - p1))) # (depth, n, code)
        nllrp = T.switch(xcode, p1, 1. - p1)  # (depth, n, code)
        nllr = -T.sum(T.log(eps + nllrp), axis=2)
        nll = T.mean(nllr, axis=None)
        loss_param_reg = T.constant(0.)
        if regularizer:
            for p in self.params:
                if p.ndim > 1:
                    loss_param_reg += regularizer(p)
        loss = nll + self.lstm.loss_activity + self.lstm.loss_temporal_activity + loss_param_reg
        updates = opt.get_updates(loss, self.params)
        self.train_fun = theano.function([self.lstm.input_x],
                                         [nll,
                                          self.lstm.loss_activity,
                                          self.lstm.loss_temporal_activity,
                                          loss_param_reg,
                                          loss],
                                         updates=updates)

        # Testing
        # old version
        """
        p1 = T.nnet.sigmoid(T.dot(self.lstm.test_y, yw) + yb)  # (depth, n, code)
        #nllrp = (xcode * T.log(eps + p1)) + ((1 - xcode) * (T.log(eps + 1. - p1)))
        nllrp = T.switch(xcode, p1, 1. - p1)
        nllr = -T.sum(T.log(eps+nllrp), axis=2)  # (depth, n)
        nll_part = T.transpose(nllr, (1, 0))  # (n, depth)
        self.nll_fun = theano.function([self.lstm.input_x], nll_part)
        """
        p1 = T.nnet.sigmoid(T.dot(self.lstm.test_y, yw) + yb)  # (depth, n, code)
        # xcode: (depth, n, code)
        # encoding: (x_k, code)
        h = (T.dot(T.log(eps + p1), T.transpose(self.encoding, (1, 0))) +
             T.dot(T.log(eps + 1. - p1), T.transpose(1 - self.encoding, (1, 0))))  # (depth, n, x_k)
        p2 = softmax_nd(h)  # (depth, n, x_k)

        mg = T.mgrid[0:p2.shape[0], 0:p2.shape[1]]
        pt = p2[mg[0], mg[1], self.lstm.xr]  # (depth, n)
        nll_part = T.transpose(-T.log(eps + pt),(1,0))
        self.nll_fun = theano.function([self.lstm.input_x], nll_part)

        train_headers = ['NLL', 'Activity Reg', 'Temporal Reg', 'Weight Reg', 'Loss']
        val_headers = ['NLL', 'PPL']
        weights = self.params + opt.weights
        super(LSTMSoftmaxSparse, self).__init__(weights=weights,
                                                train_headers=train_headers,
                                                val_headers=val_headers)

    def save_output(self, output_path, epoch, xvalid, xtest):
        """
        samples = 64
        depth = 35
        x = self.gen_fun(samples, depth)
        with open('{}/generated-{:08d}.txt'.format(output_path, epoch), 'w') as f:
            for i in range(x.shape[0]):
                s = []
                for j in range(x.shape[1]):
                    s.append(self.vocab[x[i, j]])
                f.write(" ".join(s) + "\n")
                """
        pass

    def train_batchx(self, x, **kwargs):
        return self.train_fun(x)
