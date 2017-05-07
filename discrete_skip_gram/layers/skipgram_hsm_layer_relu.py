import theano
import theano.tensor as T
from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer

from .utils import W, shift_tensor, embedding, b
from ..units.dense_unit import DenseUnit
from ..units.mlp_unit import MLPUnit


class SkipgramHSMLayerRelu(Layer):
    """
    Given a flattened representation of x, encode as a series of symbols
    """

    def __init__(self,
                 units,
                 mean=True,
                 inner_activation=T.nnet.relu,
                 embeddings_initializer='RandomUniform', embeddings_regularizer=None,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='random_uniform', bias_regularizer=None):
        self.units = units
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
        self.outer_h0 = b(self, (1, self.units), "outer_h0")
        embedding_h0 = b(self, (1, self.units), "embedding_h0")
        prediction_h0 = b(self, (1, self.units), "prediction_h0")
        y_embedding = embedding(self, (3, self.units), "y_embedding")
        self.embedding_rnn = MLPUnit(self,
                                     input_units=[self.units, self.units],
                                     units=self.units,
                                     output_units=self.units,
                                     inner_activation=self.inner_activation,
                                     name="embedding_rnn")

        embedding_params = ([y_embedding] +
                            self.embedding_rnn.non_sequences)
        self.embedding_count = len(embedding_params)
        self.embedding_mlp = MLPUnit(self,
                                     input_units=[self.units],
                                     units=self.units,
                                     output_units=self.units,
                                     inner_activation=self.inner_activation,
                                     name="embedding_mlp")

        # Main lstm
        self.outer_rnn = MLPUnit(self,
                                 input_units=[self.units, self.units, input_dim],
                                 units=self.units,
                                 output_units=self.units,
                                 inner_activation=self.inner_activation,
                                 name="outer_rnn")
        self.outer_mlp = MLPUnit(self,
                                 input_units=[self.units],
                                 units=self.units,
                                 output_units=self.units,
                                 inner_activation=self.inner_activation,
                                 name="outer_mlp")

        # Prediction step
        yp_embedding = embedding(self, (3, self.units), "yp_embedding")
        self.prediction_rnn = MLPUnit(self,
                                      input_units=[self.units, self.units, self.units],
                                      units=self.units,
                                      output_units=self.units,
                                      inner_activation=self.inner_activation,
                                      name="prediction_rnn")
        self.prediction_mlp = MLPUnit(self,
                                      input_units=[self.units],
                                      units=self.units,
                                      output_units=1,
                                      inner_activation=self.inner_activation,
                                      output_activation=T.nnet.sigmoid,
                                      name="prediction_mlp")
        prediction_params = ([yp_embedding] +
                             self.prediction_rnn.non_sequences +
                             self.prediction_mlp.non_sequences)
        self.prediction_count = len(prediction_params)

        self.non_sequences = ([embedding_h0, prediction_h0] +
                              embedding_params +
                              self.embedding_mlp.non_sequences +
                              self.outer_rnn.non_sequences +
                              self.outer_mlp.non_sequences +
                              prediction_params)
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
        rnnparams = params[idx:(idx + self.embedding_rnn.count)]
        idx += self.embedding_rnn.count
        assert idx == len(params)

        yh = y_embedding[y, :]
        hd = self.embedding_rnn.call([h0, yh], rnnparams)
        h1 = h0 + hd
        return h1

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
        embedding_h0 = params[idx]
        idx += 1
        prediction_h0 = params[idx]
        idx += 1
        embedding_params = params[idx:(idx + self.embedding_count)]
        idx += self.embedding_count
        embedding_mlp_params = params[idx:(idx + self.embedding_mlp.count)]
        idx += self.embedding_mlp.count
        outer_rnn_params = params[idx:(idx + self.outer_rnn.count)]
        idx += self.outer_rnn.count
        outer_mlp_params = params[idx:(idx + self.outer_mlp.count)]
        idx += self.outer_mlp.count
        prediction_params = params[idx:(idx + self.prediction_count)]
        idx += self.prediction_count
        assert (idx == len(params))

        n = y0.shape[1]  # (code depth, n)

        # Run embedding lstm over input embeddings
        outputs_info = [
            T.extra_ops.repeat(embedding_h0, n, axis=0)
        ]
        ytmp, _ = theano.scan(self.step_embedding, sequences=[y0], outputs_info=outputs_info,
                              non_sequences=embedding_params)
        # ytmp: (code depth, n, units)
        ytmp = ytmp[-1, :, :]  # (n, units)
        yembedded = self.embedding_mlp.call([ytmp], embedding_mlp_params)
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

    def call(self, (z, y)):
        return self.call_on_params(z, y, self.outer_h0, self.non_sequences)

    def call_on_params(self, z, y, h0, non_sequences):
        # z: input context: n, input_dim
        # y: ngram encoded: n, y depth, hsm depth int32
        yr = T.transpose(y, (1, 2, 0))  # (y depth, hsm depth, n)
        ys = shift_tensor(y)
        ysr = T.transpose(ys, (1, 2, 0))  # (y depth, hsm depth, n)
        n = z.shape[0]
        outputs_info = [T.extra_ops.repeat(h0, n, axis=0),
                        None]
        (hr, nllr), _ = theano.scan(self.step, sequences=[ysr, yr], outputs_info=outputs_info,
                                    non_sequences=[z] + list(non_sequences))
        nll = T.transpose(nllr, (1, 0))
        if self.mean:
            nll = T.mean(nll, axis=1, keepdims=True)
        return nll


class SkipgramHSMPolicyLayerRelu(Layer):
    """
    Generates encodings based on some context z
    """

    def __init__(self, layer, srng, code_depth, y_depth):
        assert isinstance(layer, SkipgramHSMLayerRelu)
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

    def step(self, rng, h0, y0, z, *params):
        print "policy Initial dims: rng {}, h0 {}, y0 {}, z {}".format(rng.ndim, h0.ndim, y0.ndim, z.ndim)
        # y0 (hsm depth, n) [0-2]
        # y1 (hsm depth, n) [0-1]
        # Unpack parameters
        idx = 0
        embedding_h0 = params[idx]
        idx += 1
        prediction_h0 = params[idx]
        idx += 1
        embedding_params = params[idx:(idx + self.layer.embedding_count)]
        idx += self.layer.embedding_count
        embedding_mlp_params = params[idx:(idx + self.layer.embedding_mlp.count)]
        idx += self.layer.embedding_mlp.count
        outer_rnn_params = params[idx:(idx + self.layer.outer_rnn.count)]
        idx += self.layer.outer_rnn.count
        outer_mlp_params = params[idx:(idx + self.layer.outer_mlp.count)]
        idx += self.layer.outer_mlp.count
        prediction_params = params[idx:(idx + self.layer.prediction_count)]
        idx += self.layer.prediction_count
        assert (idx == len(params))

        n = y0.shape[1]  # (code depth, n)

        # Run embedding lstm over input embeddings
        outputs_info = [
            T.extra_ops.repeat(embedding_h0, n, axis=0)
        ]
        ytmp, _ = theano.scan(self.layer.step_embedding, sequences=[y0], outputs_info=outputs_info,
                              non_sequences=embedding_params)
        # ytmp: (code depth, n, units)
        ytmp = ytmp[-1, :, :]  # (n, units)
        yembedded = self.layer.embedding_mlp.call([ytmp], embedding_mlp_params)
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
        print "policy Ending dims: h1 {}, ypred {}".format(h1.ndim, ypred.ndim)
        return h1, ypred

    def call(self, z):
        # z: input context: n, input_dim
        # output: n, y_depth, code_depth
        n = z.shape[0]
        rng = self.srng.uniform(low=0, high=1, dtype='float32', size=(self.y_depth, self.code_depth, n))
        outputs_info = [T.extra_ops.repeat(self.layer.outer_h0, n, axis=0),
                        T.zeros((self.code_depth, n), dtype='int32')]
        (hr, yr), _ = theano.scan(self.step, sequences=[rng], outputs_info=outputs_info,
                                  non_sequences=[z] + self.layer.non_sequences)
        # yr: (y_depth, code_depth, n)
        y = T.transpose(yr, (2, 0, 1)) - 1
        return y
