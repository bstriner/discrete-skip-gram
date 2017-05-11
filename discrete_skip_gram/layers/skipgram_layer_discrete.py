import numpy as np
import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import W, b, pair, shift_tensor, embedding, leaky_relu
from .utils import softmax_nd
from ..units.mlp_unit import MLPUnit
from ..units.lstm_muli_unit import LSTMMultiUnit
from .utils import leaky_relu


class SkipgramLayerDiscrete(Layer):
    def __init__(self,
                 z_k,
                 y_k,
                 units,
                 embedding_units,
                 hidden_layers=2,
                 inner_activation=leaky_relu,
                 mean=True,
                 srng=None,
                 negative_sampling=None,
                 embeddings_initializer='random_uniform', embeddings_regularizer=None,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.srng = srng
        self.negative_sampling = negative_sampling
        self.z_k = z_k
        self.y_k = y_k
        self.hidden_layers = hidden_layers
        self.inner_activation = inner_activation
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

    def build(self, (z, y)):
        assert (len(z) == 3)  # n, z_depth, units
        assert (len(y) == 2)  # n, y_depth
        input_dim = z[2]
        y_embedding = embedding(self, (self.y_k + 1, self.embedding_units), "y_embedding")
        self.rnn = MLPUnit(self,
                           input_units=[self.units, self.embedding_units],
                           units=self.units,
                           output_units=self.units,
                           inner_activation=self.inner_activation,
                           hidden_layers=self.hidden_layers,
                           name="rnn")
        self.mlp = MLPUnit(self,
                           input_units=[self.units, input_dim],
                           units=self.units,
                           output_units=self.y_k * self.z_k,
                           inner_activation=self.inner_activation,
                           hidden_layers=self.hidden_layers,
                           name="mlp")
        self.non_sequences = [y_embedding] + self.rnn.non_sequences + self.mlp.non_sequences
        self.h0 = b(self, (1, self.units), name="h0")
        self.built = True

    def compute_output_shape(self, (z, y)):
        assert (len(z) == 3)  # n, z_depth, units
        assert (len(y) == 2)  # n, y_depth
        if self.mean:
            return y[0], z[1], self.z_k
        else:
            return y[0], z[1], self.z_k, y[1]

    def step(self, y0, y1, h0, z, *params):
        print "Dtypes: {}, {}, {}, {}".format(y0.dtype, y1.dtype, h0.dtype, z.dtype)
        idx = 0
        y_embedding = params[idx]
        idx += 1
        rnnparams = params[idx:(idx + self.rnn.count)]
        idx += self.rnn.count
        mlpparams = params[idx:(idx + self.mlp.count)]
        idx += self.mlp.count
        assert idx == len(params)

        embedded = y_embedding[y0, :]
        hd = self.rnn.call([h0, embedded], rnnparams)
        h1 = hd + h0
        n = y0.shape[0]
        z_depth = z.shape[1]
        raw1 = self.mlp.call([h1.dimshuffle((0, 'x', 1)), z], mlpparams)  # n, z_depth, z_k*y_k
        raw2 = T.reshape(raw1, (n, z_depth, self.z_k, self.y_k))
        p1 = softmax_nd(raw2)  # n, z_depth, z_k, y_k
        eps = 1e-6
        nll1 = -T.log(eps+p1[T.arange(p1.shape[0]), :, :, y1])  # n, z_depth, z_k
        return h1, nll1

    def call(self, (z, y)):
        # z: input context: n, z_depth, input_dim
        # y: ngram: n, depth int32
        yr = T.transpose(y, (1, 0))
        yshifted = shift_tensor(y)
        yshiftedr = T.transpose(yshifted, (1, 0))
        n = z.shape[0]
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0),
                        None]
        (hr, nllr), _ = theano.scan(self.step, sequences=[yshiftedr, yr], outputs_info=outputs_info,
                                    non_sequences=[z] + self.non_sequences)
        # nllr: y_depth, n, z_depth, z_k
        nll = T.transpose(nllr, (1, 2, 3, 0))  # n, z_depth, z_k, y_depth
        if self.mean:
            nll = T.mean(nll, axis=3)
        return nll


class SkipgramPolicyLayerDiscrete(Layer):
    def __init__(self, layer, srng, depth):
        assert isinstance(layer, SkipgramLayerDiscrete)
        self.layer = layer
        self.srng = srng
        self.depth = depth
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.built = True

    def compute_output_shape(self, (zh, z)):
        assert len(zh) == 2
        assert len(z) == 2
        return z[0], self.depth

    def step(self, rng, h0, y0, zh, z, *params):
        idx = 0
        y_embedding = params[idx]
        idx += 1
        rnnparams = params[idx:(idx + self.layer.rnn.count)]
        idx += self.layer.rnn.count
        mlpparams = params[idx:(idx + self.layer.mlp.count)]
        idx += self.layer.mlp.count
        assert idx == len(params)

        embedded = y_embedding[y0, :]
        hd = self.layer.rnn.call([h0, embedded], rnnparams)
        h1 = hd + h0
        n = y0.shape[0]
        raw1 = self.layer.mlp.call([h1, zh], mlpparams)  # n, z_k*y_k
        raw2 = T.reshape(raw1, (n, self.layer.z_k, self.layer.y_k))
        p1 = softmax_nd(raw2)  # n, z_k, y_k
        p2 = p1[T.arange(p1.shape[0]), T.flatten(z), :] # n, y_k
        c1 = T.cumsum(p2, axis=1)
        y1 = T.sum(T.gt(rng.dimshuffle((0, 'x')), c1), axis=1) + 1
        y1 = T.cast(y1, 'int32')
        return h1, y1

    def call(self, (zh, z)):
        # zh: input context: n, input_dim
        # z: encoding: n, 1
        n = z.shape[0]
        rngr = self.srng.uniform(low=0, high=1, dtype='float32', size=(self.depth, n))
        outputs_info = [T.extra_ops.repeat(self.layer.h0, n, axis=0),
                        T.zeros((n,), dtype='int32')]
        (hr, yr), _ = theano.scan(self.step, sequences=[rngr], outputs_info=outputs_info,
                                  non_sequences=[zh, z] + self.layer.non_sequences)
        y = T.transpose(yr, (1, 0)) - 1
        return y
