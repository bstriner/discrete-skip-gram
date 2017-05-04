import theano
import theano.tensor as T
from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer

from .utils import W, shift_tensor, embedding
from ..units.dense_unit import DenseUnit
from ..units.lstm_unit import LSTMUnit


class SkipgramHSMLayer(Layer):
    """
    Given a flattened representation of x, encode as a series of symbols
    """

    def __init__(self,
                 units,
                 mean=True,
                 embeddings_initializer='RandomUniform',embeddings_regularizer=None,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.units = units
        self.mean = mean
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]
        self.supports_masking = False
        Layer.__init__(self)

    def build_params(self, input_dim):

        # Embedding step
        y_embedding = embedding(self, (3, self.units), "y_embedding")
        self.embedding_lstm = LSTMUnit(self, self.units, "embedding_lstm")
        embedding_params = ([y_embedding] +
                            self.embedding_lstm.non_sequences)
        self.embedding_count = len(embedding_params)

        self.embedding_d1 = DenseUnit(self, self.units, self.units, "embedding_d1", activation=T.tanh)
        self.embedding_d2 = DenseUnit(self, self.units, self.units, "embedding_d2")

        # Prediction step
        yp_embedding = embedding(self, (3, self.units), "yp_embedding")
        self.prediction_lstm = LSTMUnit(self, self.units, "prediction_lstm")
        self.prediction_d1 = DenseUnit(self, self.units, self.units, "prediction_d1", activation=T.tanh)
        self.prediction_d2 = DenseUnit(self, self.units, 1, "prediction_d2", activation=T.nnet.sigmoid)
        prediction_params = ([yp_embedding] +
                             self.prediction_lstm.non_sequences +
                             self.prediction_d1.non_sequences +
                             self.prediction_d2.non_sequences)
        self.prediction_count = len(prediction_params)

        # Main lstm
        z_W = W(self, (input_dim, self.units), "z_W")
        self.outer_lstm = LSTMUnit(self, self.units, "outer_lstm")
        self.outer_d1 = DenseUnit(self, self.units, self.units, "outer_d1", activation=T.tanh)
        self.outer_d2 = DenseUnit(self, self.units, self.units, "outer_d2")

        self.non_sequences = ([
                                  self.embedding_lstm.h0,
                                  self.prediction_lstm.h0] +
                              embedding_params +
                              self.embedding_d1.non_sequences +
                              self.embedding_d2.non_sequences +
                              prediction_params +
                              [z_W] +
                              self.outer_lstm.non_sequences +
                              self.outer_d1.non_sequences +
                              self.outer_d2.non_sequences)
        self.built = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        y = input_shape[1]
        assert (len(z) == 2)
        assert (len(y) == 3)
        input_dim = z[1]
        self.build_params(input_dim)

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        y = input_shape[1]
        assert (len(z) == 2)
        assert (len(y) == 3)  # (n, y depth, code depth)
        if self.mean:
            return y[0], 1
        else:
            return y[0], y[1]

    def step_embedding(self, y, h0, *params):
        idx = 0
        y_embedding = params[idx]
        idx += 1
        lstmparams = params[idx:(idx + self.embedding_lstm.count)]
        idx += self.embedding_lstm.count
        assert idx == len(params)

        yh = y_embedding[y, :]
        h1, o1 = self.embedding_lstm.call(h0, yh, lstmparams)
        return h1, o1

    def step_predict(self, yp0, yp1, h0, ctx, *params):
        idx = 0
        yp_embedding = params[idx]
        idx += 1
        lstmparams = params[idx:(idx + self.prediction_lstm.count)]
        idx += self.prediction_lstm.count
        d1params = params[idx:(idx + 2)]
        idx += 2
        d2params = params[idx:(idx + 2)]
        idx += 2
        assert idx == len(params)

        yh = yp_embedding[yp0, :]
        h1, o1 = self.prediction_lstm.call(h0, yh + ctx, lstmparams)
        t1 = self.prediction_d1.call(o1, d1params)
        p1 = self.prediction_d2.call(t1, d2params)
        sign = (yp1 * 2) - 1
        nll = -T.log((1 - yp1) + (sign * T.flatten(p1)))
        return h1, nll

    def step(self, y0, y1, h0, z, *params):
        print "Initial dims: y0 {}, y1 {}, h0 {}, z {}".format(y0.ndim, y1.ndim, h0.ndim, z.ndim)
        # y0 (hsm depth, n) [0-2]
        # y1 (hsm depth, n) [0-1]
        # Unpack parameters
        idx = 0
        embedding_h0 = params[idx]
        idx += 1
        prediction_h0 = params[idx]
        idx += 1
        embedding_params = params[idx:(idx + self.embedding_count)]
        idx += self.embedding_count
        embedding_d1_params = params[idx:(idx + 2)]
        idx += 2
        embedding_d2_params = params[idx:(idx + 2)]
        idx += 2
        prediction_params = params[idx:(idx + self.prediction_count)]
        idx += self.prediction_count
        z_W = params[idx]
        idx += 1
        outer_lstm_params = params[idx:(idx + self.outer_lstm.count)]
        idx += self.outer_lstm.count
        d1params = params[idx:(idx + 2)]
        idx += 2
        d2params = params[idx:(idx + 2)]
        idx += 2
        assert (idx == len(params))

        n = y0.shape[1]  # (code depth, n)

        # Run embedding lstm over input embeddings
        outputs_info = [
            T.extra_ops.repeat(embedding_h0, n, axis=0),
            None
        ]
        (_, ytmp), _ = theano.scan(self.step_embedding, sequences=[y0], outputs_info=outputs_info,
                                   non_sequences=embedding_params)
        ytmp = ytmp[-1, :, :]  # (n, units)
        ytmp = self.embedding_d1.call(ytmp, embedding_d1_params)
        yembedded = self.embedding_d2.call(ytmp, embedding_d2_params)
        print "y embedded: {}".format(yembedded.ndim)

        # Embedding of Z
        zembedded = T.dot(z, z_W)  # (n, units)

        # Run main lstm
        h1, o1 = self.outer_lstm.call(h0, yembedded + zembedded, outer_lstm_params)
        tmp = self.outer_d1.call(o1, d1params)
        ctx = self.outer_d2.call(tmp, d2params)

        # Run prediction lstm
        # y1: (code_depth, n) [0-1]
        y1s = T.concatenate((T.zeros_like(y1[0:1, :]), y1[:-1, :]), axis=0)
        outputs_info = [
            T.extra_ops.repeat(prediction_h0, n, axis=0),
            None
        ]
        (_, nllparts), _ = theano.scan(self.step_predict, sequences=[y1s, y1], outputs_info=outputs_info,
                                       non_sequences=(ctx,) + prediction_params)
        # nll (coding depth, n)
        nll = T.sum(nllparts, axis=0)  # (n,)
        print "Ending dims: h1 {}, nll {}".format(h1.ndim, nll.ndim)
        return h1, nll

    def call(self, (z, y)):
        # z: input context: n, input_dim
        # y: ngram encoded: n, y depth, hsm depth int32
        yr = T.transpose(y, (1, 2, 0))  # (y depth, hsm depth, n)
        ys = shift_tensor(y)
        ysr = T.transpose(ys, (1, 2, 0))  # (y depth, hsm depth, n)
        n = z.shape[0]
        outputs_info = [T.extra_ops.repeat(self.outer_lstm.h0, n, axis=0),
                        None]
        (hr, nllr), _ = theano.scan(self.step, sequences=[ysr, yr], outputs_info=outputs_info,
                                    non_sequences=[z] + self.non_sequences)
        nll = T.transpose(nllr, (1, 0))
        if self.mean:
            nll = T.mean(nll, axis=1, keepdims=True)
        return nll


class SkipgramHSMPolicyLayer(Layer):
    """
    Generates encodings based on some context z
    """

    def __init__(self, layer, srng, code_depth, y_depth):
        self.layer = layer
        self.srng = srng
        self.code_depth = code_depth
        self.y_depth = y_depth
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.built = True

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return input_shape[0], self.y_depth, self.code_depth

    def step_predict(self, rng, h0, yp0, ctx, *params):
        # policy
        idx = 0
        yp_embedding = params[idx]
        idx += 1
        lstmparams = params[idx:(idx + self.layer.prediction_lstm.count)]
        idx += self.layer.prediction_lstm.count
        d1params = params[idx:(idx + 2)]
        idx += 2
        d2params = params[idx:(idx + 2)]
        idx += 2
        assert idx == len(params)

        yh = yp_embedding[yp0, :]
        h1, o1 = self.layer.prediction_lstm.call(h0, yh + ctx, lstmparams)
        t1 = self.layer.prediction_d1.call(o1, d1params)
        p1 = self.layer.prediction_d2.call(t1, d2params)
        y1 = T.cast(T.flatten(p1) > rng, "int32") + 1
        return h1, y1

    def step(self, rng, h0, y0, z, *params):
        print "policy Initial dims: rng {}, h0 {}, y0 {}, z {}".format(rng.ndim, h0.ndim, y0.ndim, z.ndim)
        # policy
        # y0 (hsm depth, n) [0-2]
        # Unpack parameters
        idx = 0
        embedding_h0 = params[idx]
        idx += 1
        prediction_h0 = params[idx]
        idx += 1
        embedding_params = params[idx:(idx + self.layer.embedding_count)]
        idx += self.layer.embedding_count
        embedding_d1_params = params[idx:(idx + 2)]
        idx += 2
        embedding_d2_params = params[idx:(idx + 2)]
        idx += 2
        prediction_params = params[idx:(idx + self.layer.prediction_count)]
        idx += self.layer.prediction_count
        z_W = params[idx]
        idx += 1
        outer_lstm_params = params[idx:(idx + self.layer.outer_lstm.count)]
        idx += self.layer.outer_lstm.count
        d1params = params[idx:(idx + 2)]
        idx += 2
        d2params = params[idx:(idx + 2)]
        idx += 2
        assert (idx == len(params))

        n = y0.shape[1]  # (code depth, n)

        # Run embedding lstm over input embeddings
        outputs_info = [
            T.extra_ops.repeat(embedding_h0, n, axis=0),
            None
        ]
        (_, ytmp), _ = theano.scan(self.layer.step_embedding, sequences=[y0], outputs_info=outputs_info,
                                   non_sequences=embedding_params)
        ytmp = ytmp[-1, :, :]  # (n, units)
        ytmp = self.layer.embedding_d1.call(ytmp, embedding_d1_params)
        yembedded = self.layer.embedding_d2.call(ytmp, embedding_d2_params)
        print "policy y embedded: {}".format(yembedded.ndim)

        # Embedding of Z
        zembedded = T.dot(z, z_W)  # (n, units)

        # Run main lstm
        h1, o1 = self.layer.outer_lstm.call(h0, yembedded + zembedded, outer_lstm_params)
        tmp = self.layer.outer_d1.call(o1, d1params)
        ctx = self.layer.outer_d2.call(tmp, d2params)

        # Run prediction lstm
        # y1: (code_depth, n) [0-1]
        outputs_info = [
            T.extra_ops.repeat(prediction_h0, n, axis=0),
            T.zeros((n,), dtype='int32')
        ]
        (_, ypred), _ = theano.scan(self.step_predict, sequences=[rng], outputs_info=outputs_info,
                                    non_sequences=(ctx,) + prediction_params)
        # ypred (coding depth, n)
        print "policy Ending dims: h1 {}, ypred {}".format(h1.ndim, ypred.ndim)
        return h1, ypred

    def call(self, z):
        # z: input context: n, input_dim
        # output: n, y_depth, code_depth
        n = z.shape[0]
        rng = self.srng.uniform(low=0, high=1, dtype='float32', size=(self.y_depth, self.code_depth, n))
        outputs_info = [T.extra_ops.repeat(self.layer.outer_lstm.h0, n, axis=0),
                        T.zeros((self.code_depth, n), dtype='int32')]
        (hr, yr), _ = theano.scan(self.step, sequences=[rng], outputs_info=outputs_info,
                                  non_sequences=[z] + self.layer.non_sequences)
        # yr: (y_depth, code_depth, n)
        y = T.transpose(yr, (2, 0, 1)) - 1
        return y
