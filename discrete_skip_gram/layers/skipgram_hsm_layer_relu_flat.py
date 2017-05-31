import theano
import theano.tensor as T
from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer

from .utils import build_kernel, shift_tensor, build_embedding, build_bias
from ..units.dense_unit import DenseUnit
from ..units.mlp_unit import MLPUnit
import keras.backend as K


class SkipgramHSMLayerReluFlat(Layer):
    """
    Given a flattened representation of x, encode as a series of symbols
    """

    def __init__(self,
                 units,
                 embedding_units,
                 k,
                 mean=True,
                 inner_activation=T.nnet.relu,
                 layernorm=False,
                 embeddings_initializer='random_uniform', embeddings_regularizer=None,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='random_uniform', bias_regularizer=None):
        self.k = k
        self.layernorm = layernorm
        self.units = units
        self.embedding_units = embedding_units
        self.mean = mean
        self.inner_activation = inner_activation
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
        self.outer_h0 = build_bias(self, (1, self.units), "outer_h0")
        prediction_h0 = build_bias(self, (1, self.units), "prediction_h0")
        y_embedding = build_embedding(self, (self.k + 1, self.embedding_units), "y_embedding")

        # Main lstm
        self.outer_rnn = MLPUnit(self,
                                 input_units=[self.units, self.embedding_units, input_dim],
                                 units=self.units,
                                 output_units=self.units,
                                 inner_activation=self.inner_activation,
                                 layernorm=self.layernorm,
                                 name="outer_rnn")
        self.outer_mlp = MLPUnit(self,
                                 input_units=[self.units],
                                 units=self.units,
                                 output_units=self.units,
                                 inner_activation=self.inner_activation,
                                 layernorm=self.layernorm,
                                 name="outer_mlp")

        # Prediction step
        yp_embedding = build_embedding(self, (3, self.units), "yp_embedding")
        self.prediction_rnn = MLPUnit(self,
                                      input_units=[self.units, self.units, self.units],
                                      units=self.units,
                                      output_units=self.units,
                                      inner_activation=self.inner_activation,
                                      layernorm=self.layernorm,
                                      name="prediction_rnn")
        self.prediction_mlp = MLPUnit(self,
                                      input_units=[self.units],
                                      units=self.units,
                                      output_units=1,
                                      inner_activation=self.inner_activation,
                                      layernorm=self.layernorm,
                                      output_activation=T.nnet.sigmoid,
                                      name="prediction_mlp")
        prediction_params = ([yp_embedding] +
                             self.prediction_rnn.non_sequences +
                             self.prediction_mlp.non_sequences)
        self.prediction_count = len(prediction_params)

        self.non_sequences = ([prediction_h0] +
                              [y_embedding] +
                              self.outer_rnn.non_sequences +
                              self.outer_mlp.non_sequences +
                              prediction_params)
        self.built = True

    def build(self, (z, y0, y1)):
        assert (len(z) == 2)
        assert (len(y0) == 2)
        assert (len(y1) == 3)
        input_dim = z[1]
        self.build_params(input_dim)

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, (z, y0, y1)):
        assert (len(z) == 2)
        assert (len(y0) == 2)  # (n, y depth)
        assert (len(y1) == 3)  # (n, y depth, code depth)
        if self.mean:
            return y0[0], 1
        else:
            return y0[0], y0[1]

    def step_predict(self, yp0, yp1, h0, ctx, *params):
        idx = 0
        yp_embedding = params[idx]
        idx += 1
        rnnparams = params[idx:(idx + self.prediction_rnn.count)]
        idx += self.prediction_rnn.count
        mlpparams = params[idx:(idx + self.prediction_mlp.count)]
        idx += self.prediction_mlp.count
        assert idx == len(params)

        yh = yp_embedding[yp0, :]
        hd = self.prediction_rnn.call([h0, yh, ctx], rnnparams)
        h1 = h0 + hd
        p1 = self.prediction_mlp.call([h1], mlpparams)
        sign = (yp1 * 2) - 1
        eps = 1e-6
        nll = -T.log(eps + (1 - yp1) + (sign * T.flatten(p1)))
        return h1, nll

    def step(self, y0, y1, h0, z, *params):
        print "Initial dims: y0 {}, y1 {}, h0 {}, z {}".format(y0.ndim, y1.ndim, h0.ndim, z.ndim)
        # y0 (hsm depth, n) [0-2]
        # y1 (hsm depth, n) [0-1]
        # Unpack parameters
        idx = 0
        prediction_h0 = params[idx]
        idx += 1
        y_embedding = params[idx]
        idx += 1
        outer_rnn_params = params[idx:(idx + self.outer_rnn.count)]
        idx += self.outer_rnn.count
        outer_mlp_params = params[idx:(idx + self.outer_mlp.count)]
        idx += self.outer_mlp.count
        prediction_params = params[idx:(idx + self.prediction_count)]
        idx += self.prediction_count
        assert (idx == len(params))

        n = y0.shape[0]  # (n,)
        yembedded = y_embedding[y0, :]
        print "y embedded: {}".format(yembedded.ndim)

        # Run main lstm
        hd = self.outer_rnn.call([h0, yembedded, z], outer_rnn_params)
        h1 = h0 + hd
        ctx = self.outer_mlp.call([h1], outer_mlp_params)

        # Run prediction lstm
        # y1: (code_depth, n) [0-1]
        y1s = T.concatenate((T.zeros_like(y1[0:1, :]), y1[:-1, :] + 1), axis=0)
        outputs_info = [
            T.extra_ops.repeat(prediction_h0, n, axis=0),
            None
        ]
        (_, nllparts), _ = theano.scan(self.step_predict, sequences=[y1s, y1], outputs_info=outputs_info,
                                       non_sequences=(ctx,) + prediction_params)
        # nll (coding depth, n)
        nll = T.sum(nllparts, axis=0)  # (n,)
        assert nll.ndim == 1
        print "Ending dims: h1 {}, nll {}".format(h1.ndim, nll.ndim)
        return h1, nll

    def call(self, (z, y0, y1)):
        return self.call_on_params(z, y0, y1, self.outer_h0, self.non_sequences)

    def call_on_params(self, z, y0, y1, h0, non_sequences):
        # z: input context: n, input_dim
        # y0: n, y depth (0-k)
        # y1: n, y depth, code depth
        y0r = T.transpose(y0, (1, 0))  # (y depth,  n)
        y1r = T.transpose(y1, (1, 2, 0))  # (y depth, code depth, n)
        n = z.shape[0]
        outputs_info = [T.extra_ops.repeat(h0, n, axis=0),
                        None]
        (hr, nllr), _ = theano.scan(self.step, sequences=[y0r, y1r], outputs_info=outputs_info,
                                    non_sequences=[z] + list(non_sequences))
        nll = T.transpose(nllr, (1, 0))
        if self.mean:
            nll = T.mean(nll, axis=1, keepdims=True)
        return nll


class SkipgramHSMPolicyLayerReluFlat(Layer):
    """
    Generates encodings based on some context z
    """

    def __init__(self, layer, srng, code_depth, y_depth, wordcodes):
        assert isinstance(layer, SkipgramHSMLayerReluFlat)
        assert layer.k == wordcodes.shape[0]
        self.layer = layer
        self.srng = srng
        self.code_depth = code_depth
        self.y_depth = y_depth
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = False
        self.wordcodes = K.variable(wordcodes, dtype='int32', name='codes')
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
        rnnparams = params[idx:(idx + self.layer.prediction_rnn.count)]
        idx += self.layer.prediction_rnn.count
        mlpparams = params[idx:(idx + self.layer.prediction_mlp.count)]
        idx += self.layer.prediction_mlp.count
        assert idx == len(params)

        yh = yp_embedding[yp0, :]
        hd = self.layer.prediction_rnn.call([h0, yh, ctx], rnnparams)
        h1 = h0 + hd
        p1 = self.layer.prediction_mlp.call([h1], mlpparams)
        y1 = T.cast(T.flatten(p1) > rng, "int32") + 1
        return h1, y1

    def step(self, rng, h0, y0, z, wordcodes, *params):
        print "policy Initial dims: rng {}, h0 {}, y0 {}, z {}".format(rng.ndim, h0.ndim, y0.ndim, z.ndim)
        idx = 0
        prediction_h0 = params[idx]
        idx += 1
        y_embedding = params[idx]
        idx += 1
        outer_rnn_params = params[idx:(idx + self.layer.outer_rnn.count)]
        idx += self.layer.outer_rnn.count
        outer_mlp_params = params[idx:(idx + self.layer.outer_mlp.count)]
        idx += self.layer.outer_mlp.count
        prediction_params = params[idx:(idx + self.layer.prediction_count)]
        idx += self.layer.prediction_count
        assert (idx == len(params))

        n = y0.shape[0]  # (n,)
        yembedded = y_embedding[y0, :]
        print "y embedded: {}".format(yembedded.ndim)

        # Run main lstm
        hd = self.layer.outer_rnn.call([h0, yembedded, z], outer_rnn_params)
        h1 = h0 + hd
        ctx = self.layer.outer_mlp.call([h1], outer_mlp_params)

        # Run prediction lstm
        # y1: (code_depth, n) [0-1]
        outputs_info = [
            T.extra_ops.repeat(prediction_h0, n, axis=0),
            T.zeros((n,), dtype='int32')
        ]
        (_, ypred), _ = theano.scan(self.step_predict, sequences=[rng], outputs_info=outputs_info,
                                    non_sequences=(ctx,) + prediction_params)
        # ypred (coding depth, n)
        ypred = ypred - 1
        vals = T.power(2, T.arange(self.code_depth - 1, -1, -1)).dimshuffle((0, 'x'))
        ynum = T.sum(vals * ypred, axis=0)  # (n,)
        switch = ynum < self.layer.k
        ynum = switch * ynum
        y1 = wordcodes[ynum] + 1
        y1 = (1 - switch) + (switch * y1)

        print "policy Ending dims: h1 {}, ypred {}".format(h1.ndim, ypred.ndim)
        return h1, y1

    def call(self, z):
        # z: input context: n, input_dim
        # output: n, y_depth, code_depth
        n = z.shape[0]
        rng = self.srng.uniform(low=0, high=1, dtype='float32', size=(self.y_depth, self.code_depth, n))
        outputs_info = [T.extra_ops.repeat(self.layer.outer_h0, n, axis=0),
                        T.zeros((n,), dtype='int32')]
        (hr, yr), _ = theano.scan(self.step, sequences=[rng], outputs_info=outputs_info,
                                  non_sequences=[z, self.wordcodes] + self.layer.non_sequences)
        # yr: (y_depth, n)
        y = T.transpose(yr, (1, 0)) - 1
        return y
