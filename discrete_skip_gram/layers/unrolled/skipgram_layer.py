import numpy as np
import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from ..utils import W, b, pair, shift_tensor, embedding, leaky_relu
from ...units.mlp_unit import MLPUnit
from ...units.lstm_muli_unit import LSTMMultiUnit
from ..utils import leaky_relu


class SkipgramLayer(Layer):
    """
    Given a flattened context, calculate NLL of a series
    """

    def __init__(self, k, units, embedding_units, mean=True,
                 srng=None,
                 negative_sampling=None,
                 embeddings_initializer='random_uniform', embeddings_regularizer=None,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.srng = srng
        self.negative_sampling = negative_sampling
        self.k = k
        self.units = units
        self.embedding_units = embedding_units
        self.mean = mean
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]
        self.supports_masking = False
        Layer.__init__(self)

    def build_params(self, input_dim):
        y_embedding = embedding(self, (self.k + 1, self.embedding_units), "h_U1")
        self.lstm = LSTMMultiUnit(self,
                                  input_units=[input_dim, self.embedding_units],
                                  units=self.units,
                                  name="lstm")
        self.mlp = MLPUnit(self, input_units=[self.units], units=self.units,
                           inner_activation=leaky_relu,
                           layernorm=True,
                           hidden_layers=3,
                           output_units=self.k,
                           output_activation=T.nnet.softmax,
                           name="mlp")
        self.non_sequences = [y_embedding] + self.lstm.non_sequences + self.mlp.non_sequences
        self.built = True

    def build(self, (z, y)):
        assert (len(z) == 2)
        assert (len(y) == 2)
        input_dim = z[1]
        self.build_params(input_dim)

    #    def compute_mask(self, inputs, mask=None):
    #        print ("Compute mask {}".format(mask))
    #        return mask

    def compute_output_shape(self, (z, y)):
        assert (len(z) == 2)
        assert (len(y) == 2)
        if self.mean:
            return y[0], 1
        else:
            return y

    def step(self, y0, y1, h0, z, *params):
        print "Dtypes: {}, {}, {}, {}".format(y0.dtype, y1.dtype, h0.dtype, z.dtype)
        idx = 0
        y_embedding = params[idx]
        idx += 1
        lstmparams = params[idx:(idx + self.lstm.count)]
        idx += self.lstm.count
        mlpparams = params[idx:(idx + self.mlp.count)]
        idx += self.mlp.count
        assert idx == len(params)

        embedded = y_embedding[y0, :]
        h1, o1 = self.lstm.call(h0, [z, embedded], lstmparams)
        p1 = self.mlp.call([o1], mlpparams)
        nll1 = -T.log(p1[T.arange(p1.shape[0]), y1])
        return h1, nll1

    def step_ns(self, y0, y1, rng, h0, z, *params):
        print "NS Dtypes: {}, {}, {}, {}, {}".format(y0.dtype, y1.dtype, rng.dtype, h0.dtype, z.dtype)
        idx = 0
        y_embedding = params[idx]
        idx += 1
        lstmparams = params[idx:idx + self.lstm.count]
        idx += self.lstm.count
        mlpparams = params[idx:idx + self.mlp.count]
        idx += self.mlp.count
        assert idx == len(params)

        embedded = y_embedding[y0, :]
        h1, o1 = self.lstm.call(h0, [z, embedded], lstmparams)
        raw1 = self.mlp.call([o1], mlpparams)
        neg = raw1[:, rng]  # (n, negative_sampling)
        scale = np.float32(self.k) / np.float32(self.negative_sampling)
        eps = 1e10
        norm = T.log((scale * T.sum(T.exp(neg), axis=1)) + eps)
        nll1 = norm - (raw1[T.arange(raw1.shape[0]), y1])
        return h1, nll1

    def call(self, (z, y)):
        # z: input context: n, input_dim
        # y: ngram: n, depth int32
        # print "Z NDIM: {}".format(z.ndim)
        yr = T.transpose(y, (1, 0))
        yshifted = shift_tensor(y)
        yshiftedr = T.transpose(yshifted, (1, 0))
        n = z.shape[0]
        outputs_info = [T.extra_ops.repeat(self.lstm.h0, n, axis=0),
                        None]
        """
        if self.negative_sampling:
            depth = y.shape[1]
            rng = self.srng.random_integers(low=0, high=self.k - 1, size=(depth, self.negative_sampling))
            (hr, nllr), _ = theano.scan(self.step_ns, sequences=[yshiftedr, yr, rng], outputs_info=outputs_info,
                                        non_sequences=[z] + self.non_sequences)
        """
        (hr, nllr), _ = theano.scan(self.step, sequences=[yshiftedr, yr], outputs_info=outputs_info,
                                        non_sequences=[z] + self.non_sequences)
        nll = T.transpose(nllr, (1, 0))
        if self.mean:
            nll = T.mean(nll, axis=1, keepdims=True)
        return nll


class SkipgramPolicyLayer(Layer):
    def __init__(self, layer, srng, depth):
        self.layer = layer
        self.srng = srng
        self.depth = depth
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.built = True

    def compute_mask(self, inputs, mask=None):
        # print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return (input_shape[0], self.depth)

    def step(self, rng, h0, y0, z, *params):
        idx = 0
        y_embedding = params[idx]
        idx += 1
        lstmparams = params[idx:idx + self.layer.lstm.count]
        idx += self.layer.lstm.count
        mlpparams = params[idx:idx + self.layer.mlp.count]
        idx += self.layer.mlp.count
        assert idx == len(params)

        embedded = y_embedding[y0, :]
        h1, o1 = self.layer.lstm.call(h0, [z, embedded], lstmparams)
        p1 = self.layer.mlp.call([o1], mlpparams)

        c1 = T.cumsum(p1, axis=1)
        y1 = T.sum(T.gt(rng.dimshuffle((0, 'x')), c1), axis=1) + 1
        y1 = T.cast(y1, 'int32')
        return h1, y1

    def call(self, z):
        # z: input context: n, input_dim
        n = z.shape[0]
        rngr = self.srng.uniform(low=0, high=1, dtype='float32', size=(self.depth, n))
        outputs_info = [T.extra_ops.repeat(self.layer.lstm.h0, n, axis=0),
                        T.zeros((n,), dtype='int32')]
        (hr, yr), _ = theano.scan(self.step, sequences=[rngr], outputs_info=outputs_info,
                                  non_sequences=[z] + self.layer.non_sequences)
        y = T.transpose(yr, (1, 0)) - 1
        return y
