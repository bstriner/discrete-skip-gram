import theano
import theano.tensor as T
from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer

from .utils import W, shift_tensor, embedding
from ..units.dense_unit import DenseUnit
from ..units.lstm_unit import LSTMUnit


class SkipgramHSMLayerFast(Layer):
    """
    Given a flattened representation of x, encode as a series of symbols
    Fast time distributed.
    """

    def __init__(self,
                 units,
                 mean=True,
                 embeddings_initializer='RandomUniform', embeddings_regularizer=None,
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
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]
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
        self.prediction_d2 = DenseUnit(self, self.units, self.units, "prediction_d2", activation=T.tanh)
        self.prediction_d3 = DenseUnit(self, self.units, self.units, "prediction_d3")
        self.prediction_y1 = DenseUnit(self, self.units, self.units, "prediction_y1", activation=T.tanh)
        self.prediction_y2 = DenseUnit(self, self.units, self.units, "prediction_y2", activation=T.tanh)
        self.prediction_y3 = DenseUnit(self, self.units, 1, "prediction_y3", activation=T.nnet.sigmoid)
        prediction_params = ([yp_embedding] +
                             self.prediction_lstm.non_sequences +
                             self.prediction_d1.non_sequences +
                             self.prediction_d2.non_sequences +
                             self.prediction_d3.non_sequences +
                             self.prediction_y1.non_sequences +
                             self.prediction_y2.non_sequences +
                             self.prediction_y3.non_sequences)
        self.prediction_count = len(prediction_params)

        # Main lstm
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
                              self.outer_lstm.non_sequences +
                              self.outer_d1.non_sequences +
                              self.outer_d2.non_sequences)
        self.built = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        y = input_shape[1]
        assert (len(z) == 3)
        assert (len(y) == 3)
        input_dim = z[2]
        self.build_params(input_dim)

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        y = input_shape[1]
        assert (len(z) == 3) # (n, z depth, input dim)
        assert (len(y) == 3)  # (n, y depth, code depth)
        if self.mean:
            return y[0], z[1]
        else:
            return y[0], z[1], y[1]

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

    def step_predict(self, yp0, yp1, h0, ctx, z, *params):
        # z (n, z depth, units)
        idx = 0
        yp_embedding = params[idx]
        idx += 1
        lstmparams = params[idx:(idx + self.prediction_lstm.count)]
        idx += self.prediction_lstm.count
        d1params = params[idx:(idx + 2)]
        idx += 2
        d2params = params[idx:(idx + 2)]
        idx += 2
        d3params = params[idx:(idx + 2)]
        idx += 2
        y1params = params[idx:(idx + 2)]
        idx += 2
        y2params = params[idx:(idx + 2)]
        idx += 2
        y3params = params[idx:(idx + 2)]
        idx += 2
        assert idx == len(params)

        yh = yp_embedding[yp0, :]
        h1, o1 = self.prediction_lstm.call(h0, yh + ctx, lstmparams)
        h = self.prediction_d1.call(o1, d1params)
        h = self.prediction_d2.call(h, d2params)
        h = self.prediction_d3.call(h, d3params) # (n, units) representation of previous words and codes
        h = z + (h.dimshuffle((0,'x',1)))
        h = self.prediction_y1.call3d(h, y1params)
        h = self.prediction_y2.call3d(h, y2params)
        yp = self.prediction_y3.call3d(h, y3params) # (n, z depth, 1)
        yp = yp[:,:,0]
        sign = (yp1 * 2) - 1 # (n,)
        nll = -T.log((1 - yp1.dimshuffle((0,'x'))) + (sign.dimshuffle((0,'x')) * yp))
        # nll: (n, z depth)
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

        # Run main lstm
        h1, o1 = self.outer_lstm.call(h0, yembedded, outer_lstm_params)
        tmp = self.outer_d1.call(o1, d1params)
        ctx = self.outer_d2.call(tmp, d2params)

        # Run prediction lstm
        # y1: (code_depth, n) [0-1]
        y1s = T.concatenate((T.zeros_like(y1[0:1, :]), y1[:-1, :]+1), axis=0)
        outputs_info = [
            T.extra_ops.repeat(prediction_h0, n, axis=0),
            None
        ]
        (_, nllparts), _ = theano.scan(self.step_predict, sequences=[y1s, y1], outputs_info=outputs_info,
                                       non_sequences=(ctx,z) + prediction_params)
        # nll (coding depth, n, z depth)
        nll = T.sum(nllparts, axis=0)  # (n, z depth)
        print "Ending dims: h1 {}, nll {}".format(h1.ndim, nll.ndim)
        return h1, nll

    def call(self, (z, y)):
        # z: input context: n, z depth, input_dim
        # y: ngram encoded: n, y depth, hsm depth int32
        yr = T.transpose(y, (1, 2, 0))  # (y depth, hsm depth, n)
        ys = shift_tensor(y)
        ysr = T.transpose(ys, (1, 2, 0))  # (y depth, hsm depth, n)
        n = z.shape[0]
        outputs_info = [T.extra_ops.repeat(self.outer_lstm.h0, n, axis=0),
                        None]
        (hr, nllr), _ = theano.scan(self.step, sequences=[ysr, yr], outputs_info=outputs_info,
                                    non_sequences=[z] + list(self.non_sequences))
        # nllr: (y depth, n, z depth)
        nll = T.transpose(nllr, (1, 2, 0))
        if self.mean:
            nll = T.mean(nll, axis=2, keepdims=False)
        return nll


class SkipgramHSMPolicyLayerFast(Layer):
    """
    Generates encodings based on some context z
    """

    def __init__(self, layer, srng, code_depth, y_depth):
        assert isinstance(layer, SkipgramHSMLayerFast)
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

    def step_predict(self, rng, h0, yp0, ctx, z, *params):
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
        d3params = params[idx:(idx + 2)]
        idx += 2
        y1params = params[idx:(idx + 2)]
        idx += 2
        y2params = params[idx:(idx + 2)]
        idx += 2
        y3params = params[idx:(idx + 2)]
        idx += 2
        assert idx == len(params)

        yh = yp_embedding[yp0, :]
        h1, o1 = self.layer.prediction_lstm.call(h0, yh + ctx, lstmparams)
        h = self.layer.prediction_d1.call(o1, d1params)
        h = self.layer.prediction_d2.call(h, d2params)
        h = self.layer.prediction_d3.call(h, d3params)  # (n, units) representation of previous words and codes
        h = z + h
        h = self.layer.prediction_y1.call(h, y1params)
        h = self.layer.prediction_y2.call(h, y2params)
        yp = self.layer.prediction_y3.call(h, y3params)  # (n, 1)
        y1 = T.cast(T.flatten(yp) > rng, "int32") + 1
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
        print "y embedded: {}".format(yembedded.ndim)

        # Run main lstm
        h1, o1 = self.layer.outer_lstm.call(h0, yembedded, outer_lstm_params)
        tmp = self.layer.outer_d1.call(o1, d1params)
        ctx = self.layer.outer_d2.call(tmp, d2params)

        # Run prediction lstm
        # y1: (code_depth, n) [0-1]
        outputs_info = [
            T.extra_ops.repeat(prediction_h0, n, axis=0),
            T.zeros((n,), dtype='int32')
        ]
        (_, ypred), _ = theano.scan(self.step_predict, sequences=[rng], outputs_info=outputs_info,
                                    non_sequences=(ctx, z) + prediction_params)
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
