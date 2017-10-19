import keras.backend as K
import theano
import theano.tensor as T

from .lstm_unit import LSTMUnit
from ..initializers import uniform_initializer


class LSTMModel(object):
    # base LSTM model
    def __init__(self,
                 x_k,
                 srng,
                 initializer=uniform_initializer(0.05),
                 units=1024,
                 layers=2,
                 activity_reg=2.,
                 temporal_activity_reg=1.,
                 input_droput=0.1,
                 zoneout=0.5,
                 output_dropout=0.5):
        self.zoneout = zoneout
        # Parameters
        xembed = K.variable(initializer((x_k + 1, units)))
        self.params = [xembed]
        self.lstms = []
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
        self.xr = xr
        self.xrs = xrs
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
        train_y = y0

        loss_activity = T.constant(0.)
        loss_temporal_activity = T.constant(0.)
        if activity_reg > 0:
            for h in y1s:
                loss_activity += activity_reg * T.mean(T.square(h), axis=None)
        if temporal_activity_reg > 0:
            for h in y1s:
                loss_temporal_activity += temporal_activity_reg * T.mean(T.square((h[1:, :, :]) - (h[:-1, :, :])),
                                                                         axis=None)

        self.input_x = input_x
        self.loss_activity = loss_activity
        self.loss_temporal_activity = loss_temporal_activity
        self.train_y = train_y

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
        test_y = y0
        self.test_y = test_y

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
